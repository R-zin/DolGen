import { Component, useEffect, useImperativeHandle, useRef, forwardRef } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { WebGLPathTracer } from 'three-gpu-pathtracer'

// three-gpu-pathtracer 0.0.24's EquirectHdrInfoUniform reads
// `scene.environment.image.{width,height,data}`, so `scene.environment` must be
// an equirectangular *DataTexture*. `pmrem.fromScene()` returns a PMREM cubeUV
// render-target texture whose `.image` is undefined — assigning it directly
// crashes setScene with "Cannot read properties of undefined (reading '0')".
// Render RoomEnvironment into a float equirect DataTexture instead.
function makeEquirectEnv(renderer, width = 512, height = 256) {
  const envScene = new RoomEnvironment()
  const pmrem = new THREE.PMREMGenerator(renderer)
  const cubeRT = pmrem.fromScene(envScene, 0.04)

  const rt = new THREE.WebGLRenderTarget(width, height, {
    type: THREE.FloatType,
    format: THREE.RGBAFormat,
    colorSpace: THREE.LinearSRGBColorSpace,
    depthBuffer: false,
  })
  const blitCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
  const blitScene = new THREE.Scene()
  blitScene.background = cubeRT.texture
  const prevRT = renderer.getRenderTarget()
  renderer.setRenderTarget(rt)
  renderer.render(blitScene, blitCam)
  renderer.setRenderTarget(prevRT)

  const data = new Float32Array(width * height * 4)
  renderer.readRenderTargetPixels(rt, 0, 0, width, height, data)

  cubeRT.dispose()
  pmrem.dispose()
  rt.dispose()

  const tex = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType)
  tex.mapping = THREE.EquirectangularReflectionMapping
  tex.needsUpdate = true
  return tex
}

/**
 * A three-gpu-pathtracer viewport.
 *
 * Renders `glb` (an ArrayBuffer) with progressive path tracing. The parent
 * controls ceiling visibility, auto-rotate, and a sample "quality" target via
 * props; render stats are reported up through onStats({samples, converged}).
 *
 * Notes vs. the old inline viewer:
 *  - the loop calls pathTracer.renderSample() every frame (it never resets each
 *    frame, so samples actually accumulate and converge),
 *  - an environment map is provided (RoomEnvironment) so the path tracer has
 *    something to light the scene with — without it the render was black,
 *  - uses the 0.0.24 API (renderSample/enablePathTracing/setScene); there is no
 *    targetSamples/visible flag on this version.
 */
const PathTracerView = forwardRef(function PathTracerView(
  { glb, ceilingVisible, autoRotate, targetSamples, onStats, onLoaded },
  ref,
) {
  const hostRef = useRef(null)
  const stateRef = useRef(null)
  // latest props the animation loop should read without re-subscribing
  const live = useRef({ ceilingVisible, autoRotate, targetSamples })
  live.current = { ceilingVisible, autoRotate, targetSamples }

  useImperativeHandle(ref, () => ({
    resetView: () => {
      const s = stateRef.current
      if (s?.model) fitCamera(s.camera, s.controls, s.model)
      s?.pathTracer.updateCamera()
    },
  }))

  // one-time renderer/scene/pathtracer setup
  useEffect(() => {
    const host = hostRef.current
    // Tag each init stage so a failure says WHICH step broke (the error
    // boundary surfaces the tag; see step() below).
    const step = (name, fn) => {
      try {
        return fn()
      } catch (e) {
        e.message = `[init:${name}] ${e.message}`
        throw e
      }
    }

    const renderer = step('renderer', () => {
      const r = new THREE.WebGLRenderer({ antialias: true })
      r.toneMapping = THREE.ACESFilmicToneMapping
      r.toneMappingExposure = 1.0
      r.outputColorSpace = THREE.SRGBColorSpace
      r.setPixelRatio(window.devicePixelRatio)
      host.appendChild(r.domElement)
      return r
    })

    let scene, envTex
    step('environment', () => {
      scene = new THREE.Scene()
      scene.background = new THREE.Color(0x1a1d22)
      // Environment lighting as an equirect DataTexture (see makeEquirectEnv).
      envTex = makeEquirectEnv(renderer)
      scene.environment = envTex
      scene.environmentIntensity = 0.9
    })

    const camera = step('camera', () => {
      const c = new THREE.PerspectiveCamera(45, 1, 0.05, 500)
      c.position.set(9, 11, 9)
      return c
    })

    const controls = step('controls', () => {
      const c = new OrbitControls(camera, renderer.domElement)
      c.enableDamping = true
      c.dampingFactor = 0.08
      c.target.set(0, 0.5, 0)
      c.autoRotateSpeed = 1.4
      return c
    })

    const pathTracer = step('pathtracer', () => {
      const pt = new WebGLPathTracer(renderer)
      // In three-gpu-pathtracer 0.0.24 `tiles` is a read-only getter returning
      // the inner renderer's Vector2 — mutate it via .set() rather than assigning.
      pt.tiles.set(3, 3)
      pt.renderScale = 0.85
      pt.dynamicLowRes = true
      pt.minSamples = 3
      return pt
    })
    step('setScene', () => pathTracer.setScene(scene, camera))

    const s = {
      renderer,
      scene,
      camera,
      controls,
      pathTracer,
      envTex,
      model: null,
      ceilingMesh: null,
      convergedLogged: false,
      disposed: false,
    }
    stateRef.current = s

    const onChange = () => {
      pathTracer.updateCamera()
      s.convergedLogged = false
    }
    controls.addEventListener('change', onChange)

    const resize = () => {
      const w = host.clientWidth
      const h = host.clientHeight
      if (!w || !h) return
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      pathTracer.setCamera(camera)
    }
    const ro = new ResizeObserver(resize)
    ro.observe(host)
    resize()

    renderer.setAnimationLoop(() => {
      if (s.disposed) return
      const L = live.current
      controls.autoRotate = L.autoRotate
      controls.update()

      if (s.model) {
        pathTracer.renderSample()
        const n = Math.floor(pathTracer.samples)
        const target = L.targetSamples
        const converged = n >= target
        if (converged && !s.convergedLogged) {
          s.convergedLogged = true
          onStats?.({ samples: n, converged, justConverged: true })
        } else {
          onStats?.({ samples: n, converged, justConverged: false })
        }
      }
    })

    return () => {
      s.disposed = true
      renderer.setAnimationLoop(null)
      ro.disconnect()
      controls.removeEventListener('change', onChange)
      controls.dispose()
      pathTracer.dispose?.()
      envTex.dispose()
      renderer.dispose()
      host.removeChild(renderer.domElement)
      stateRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // react to quality-target changes: reset accumulation so it re-converges
  useEffect(() => {
    const s = stateRef.current
    if (s) {
      s.pathTracer.reset()
      s.convergedLogged = false
    }
  }, [targetSamples])

  // ceiling toggle
  useEffect(() => {
    const s = stateRef.current
    if (!s) return
    if (s.ceilingMesh) {
      s.ceilingMesh.visible = ceilingVisible
      s.pathTracer.updateScene?.()
      s.pathTracer.reset()
      s.convergedLogged = false
    }
  }, [ceilingVisible])

  // load a new GLB
  useEffect(() => {
    const s = stateRef.current
    if (!s || !glb) return
    let cancelled = false
    ;(async () => {
      try {
        const gltf = await new GLTFLoader().parseAsync(glb.slice(0), '')
        if (cancelled) return
        // drop the previous model
        if (s.model) {
          s.scene.remove(s.model)
          disposeObject(s.model)
        }
        s.model = gltf.scene
        s.scene.add(gltf.scene)

        s.ceilingMesh = null
        gltf.scene.traverse((o) => {
          if (!s.ceilingMesh && o.name === 'ceiling') s.ceilingMesh = o
        })
        if (s.ceilingMesh) s.ceilingMesh.visible = live.current.ceilingVisible

        fitCamera(s.camera, s.controls, gltf.scene)
        // Rebuild the BVH against a wrapper scene that contains the model.
        s.pathTracer.setScene(s.scene, s.camera)
        s.pathTracer.reset()
        s.convergedLogged = false
        onLoaded?.({ hasCeiling: !!s.ceilingMesh })
      } catch (err) {
        onLoaded?.({ error: err })
      }
    })()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [glb])

  return (
    <div ref={hostRef} className="view-host" style={{ position: 'absolute', inset: 0 }} />
  )
})

function fitCamera(camera, controls, obj) {
  const box = new THREE.Box3().setFromObject(obj)
  const center = box.getCenter(new THREE.Vector3())
  const sphere = box.getBoundingSphere(new THREE.Sphere())
  const d = Math.max(sphere.radius * 2.4, 4)
  camera.position.set(center.x + d * 0.75, d * 0.85, center.z + d * 0.75)
  controls.target.copy(center).setY(Math.min(center.y, 0.8))
  camera.updateProjectionMatrix()
}

function disposeObject(root) {
  root.traverse((o) => {
    if (o.geometry) o.geometry.dispose()
    const mats = Array.isArray(o.material) ? o.material : [o.material]
    mats.forEach((m) => {
      if (!m) return
      for (const k in m) {
        const v = m[k]
        if (v && v.isTexture) v.dispose()
      }
      m.dispose()
    })
  })
}

// A WebGL/path-tracer init failure (GPU blocklisted, headless browser, old
// three-gpu-pathtracer on this device) must not take the whole app down with
// it — without this boundary the error bubbles to the root and the page goes
// black, panel included.
class PathTracerErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }
  static getDerivedStateFromError(error) {
    return { error }
  }
  componentDidCatch(error, info) {
    console.error('PathTracerView crashed:', error)
    console.error('PathTracerView error stack:', error?.stack)
    console.error('PathTracerView component stack:', info?.componentStack)
  }
  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ff7b72',
            fontSize: 13,
            textAlign: 'center',
            padding: 24,
          }}
        >
          <div style={{ fontSize: 32 }}>⚠</div>
          <div>The 3D viewer failed to start (WebGL unavailable?).</div>
          <div style={{ color: '#8f98a5', fontFamily: 'monospace', fontSize: 11.5 }}>
            {String(this.state.error?.message || this.state.error)}
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

const PathTracerViewWithBoundary = forwardRef(function PathTracerViewWithBoundary(props, ref) {
  return (
    <PathTracerErrorBoundary>
      <PathTracerView ref={ref} {...props} />
    </PathTracerErrorBoundary>
  )
})

export default PathTracerViewWithBoundary
