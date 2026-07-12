const container = document.getElementById('vizContainer');
const btnNext = document.getElementById('btnNext');
const btnPrev = document.getElementById('btnPrev');
const statusOverlay = document.getElementById('statusOverlay');
const explanationText = document.getElementById('explanationText');
const toggleOutliers = document.getElementById('toggleOutliers');
const numRaysSlider = document.getElementById('numRaysSlider');
const numRaysDisplay = document.getElementById('numRaysDisplay');
const pointSizeSlider = document.getElementById('pointSizeSlider');
const pointSizeDisplay = document.getElementById('pointSizeDisplay');

const valFront = document.getElementById('valFront');
const valBack = document.getElementById('valBack');
const valSDF = document.getElementById('valSDF');
const monitorStatus = document.getElementById('monitorStatus');

// --- THREE.JS SETUP ---
const scene = new THREE.Scene();
scene.background = null; 

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(25, 20, 35);
const initialCameraPos = camera.position.clone();
const initialCameraQuat = camera.quaternion.clone();

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.localClippingEnabled = true;
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
let controlsEnabled = true;

// Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

// --- GEOMETRY ---
const torusGeometry = new THREE.TorusGeometry(8, 3, 32, 64);
const torusMaterial = new THREE.MeshPhongMaterial({
    color: 0x0ea5e9,
    transparent: true,
    opacity: 0.35,
    wireframe: true,
    side: THREE.DoubleSide
});
const torus = new THREE.Mesh(torusGeometry, torusMaterial);
torus.rotation.x = Math.PI / 2;
scene.add(torus);
torus.updateMatrixWorld(true);

// Highlighted Triangle Group
let activeTriangleMesh = null;
let activeTriangleGeo = null;

let originalIndices = null;
let triangleLayers = [];

function computePeelingLayers() {
    if (!originalIndices) {
        originalIndices = torusGeometry.index.clone();
    }
    // Restore full geometry for accurate raycasting
    torus.geometry.setIndex(originalIndices);
    
    triangleLayers = [];
    const pos = torusGeometry.attributes.position;
    const idx = originalIndices;
    const numFaces = idx.count / 3;
    
    const camDir = new THREE.Vector3(0,0,-1).applyQuaternion(selectedDepthCamera.quaternion).normalize();
    const localRay = new THREE.Raycaster();
    
    for (let i = 0; i < numFaces; i++) {
        let vA = new THREE.Vector3().fromBufferAttribute(pos, idx.getX(i*3)).applyMatrix4(torus.matrixWorld);
        let vB = new THREE.Vector3().fromBufferAttribute(pos, idx.getY(i*3)).applyMatrix4(torus.matrixWorld);
        let vC = new THREE.Vector3().fromBufferAttribute(pos, idx.getZ(i*3)).applyMatrix4(torus.matrixWorld);
        
        let centroid = new THREE.Vector3().add(vA).add(vB).add(vC).divideScalar(3);
        let localCentroid = centroid.clone().applyMatrix4(selectedDepthCamera.matrixWorldInverse);
        
        // Orthographic bounds of our miniCam
        if (localCentroid.x < -10 || localCentroid.x > 10 || localCentroid.y < -10 || localCentroid.y > 10) {
            triangleLayers.push(0);
            continue;
        }
        
        let rayOriginLocal = new THREE.Vector3(localCentroid.x, localCentroid.y, 0);
        let rayOriginWorld = rayOriginLocal.applyMatrix4(selectedDepthCamera.matrixWorld);
        
        localRay.set(rayOriginWorld, camDir);
        let intersects = localRay.intersectObject(torus);
        
        let layer = 0;
        for (let j = 0; j < intersects.length; j++) {
            if (intersects[j].faceIndex === i) {
                layer = j + 1;
                break;
            }
        }
        
        // Fallback for floating point misses
        if (layer === 0 && intersects.length > 0) {
            let minDist = Infinity;
            let bestLayer = 0;
            for (let j = 0; j < intersects.length; j++) {
                let dist = intersects[j].point.distanceTo(centroid);
                if (dist < minDist) {
                    minDist = dist;
                    bestLayer = j + 1;
                }
            }
            if (minDist < 1.0) {
                layer = bestLayer;
            }
        }
        
        triangleLayers.push(layer);
    }
}

function applyPeelLevel(level) {
    if (!originalIndices) return;
    if (level === 0) {
        torus.geometry.setIndex(originalIndices);
        return;
    }
    const newIndices = [];
    const idx = originalIndices;
    for (let i = 0; i < triangleLayers.length; i++) {
        let l = triangleLayers[i];
        // Keep triangle if it's outside viewport (l=0) or its layer is strictly deeper than the peeling level
        if (l === 0 || l > level) { 
            newIndices.push(idx.getX(i*3), idx.getY(i*3), idx.getZ(i*3));
        }
    }
    torus.geometry.setIndex(newIndices);
    torus.geometry.computeVertexNormals();
}

// --- DEPTH MONITOR & ORTHOGRAPHIC CAMERA ---
const depthContainer = document.getElementById('depthCanvasContainer');
let selectedDepthCamera = new THREE.OrthographicCamera(-10, 10, 10, -10, 5, 35);
selectedDepthCamera.position.set(0, 0, 20); 
selectedDepthCamera.lookAt(0,0,0);

const depthRenderer = new THREE.WebGLRenderer({ antialias: false });
depthRenderer.setSize(256, 256);
depthRenderer.localClippingEnabled = true;
depthContainer.appendChild(depthRenderer.domElement);

const depthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.BasicDepthPacking, side: THREE.DoubleSide });

// Custom depth material to match calcColor exactly in the main viewport
const customDepthMaterial = new THREE.ShaderMaterial({
    uniforms: THREE.UniformsUtils.merge([
        THREE.UniformsLib.clipping,
        { camPos: { value: new THREE.Vector3() } }
    ]),
    clipping: true,
    vertexShader: `
        #include <common>
        #include <clipping_planes_pars_vertex>
        varying vec3 vWorldPosition;
        void main() {
            vec4 worldPosition = modelMatrix * vec4(position, 1.0);
            vWorldPosition = worldPosition.xyz;
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_Position = projectionMatrix * mvPosition;
            #include <clipping_planes_vertex>
        }
    `,
    fragmentShader: `
        #include <common>
        #include <clipping_planes_pars_fragment>
        uniform vec3 camPos;
        varying vec3 vWorldPosition;
        void main() {
            #include <clipping_planes_fragment>
            float dist = distance(camPos, vWorldPosition);
            float intensity = 1.0 - ((dist - 9.0) / 11.0);
            intensity = clamp(intensity, 0.0, 1.0);
            gl_FragColor = vec4(intensity, intensity, intensity, 1.0);
        }
    `,
    side: THREE.DoubleSide
});

const torusDepth = new THREE.Mesh(torusGeometry, depthMaterial);
torusDepth.rotation.x = Math.PI / 2;

const hitSpheresGroup = new THREE.Group();
scene.add(hitSpheresGroup);

let currentIntersects = [];
let hitSpheres = [];

// --- MULTIPLE RAYS (Fibonacci Sphere) & INTERACTION ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let cameraObjects = [];
let activeCameraObj = null;

const camGeo = new THREE.ConeGeometry(1.5, 3.0, 4);
camGeo.rotateX(Math.PI / 2);
const camMatNormal = new THREE.MeshBasicMaterial({ color: 0xfacc15, wireframe: false });
const camMatActive = new THREE.MeshBasicMaterial({ color: 0xf43f5e, wireframe: false });

function updateMultiRays(numRays) {
    cameraObjects.forEach(obj => {
        scene.remove(obj.mesh);
        scene.remove(obj.helper);
    });
    cameraObjects = [];
    activeCameraObj = null;
    
    if (numRays < 1) return;

    const offset = 2 / numRays;
    const increment = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < numRays; i++) {
        const y = ((i * offset) - 1) + (offset / 2);
        const r = Math.sqrt(1 - Math.pow(y, 2));
        const phi = i * increment;
        const x = Math.cos(phi) * r;
        const z = Math.sin(phi) * r;
        
        let dir = new THREE.Vector3(x, y, z).normalize();
        let pos = dir.clone().multiplyScalar(20);
        
        let miniCam = new THREE.OrthographicCamera(-10, 10, 10, -10, 1, 40);
        miniCam.position.copy(pos);
        miniCam.lookAt(0,0,0);
        miniCam.updateMatrixWorld();
        
        let helper = new THREE.CameraHelper(miniCam);
        scene.add(helper);
        
        let mesh = new THREE.Mesh(camGeo, camMatNormal);
        mesh.position.copy(pos);
        mesh.lookAt(0,0,0);
        
        let obj = { mesh: mesh, helper: helper, camera: miniCam, index: i };
        mesh.userData = obj;
        scene.add(mesh);
        
        cameraObjects.push(obj);
    }
    
    if (cameraObjects.length > 0) {
        selectCamera(cameraObjects[0]);
    }
}

function selectCamera(camObj) {
    if (activeCameraObj) {
        activeCameraObj.mesh.material = camMatNormal;
    }
    activeCameraObj = camObj;
    activeCameraObj.mesh.material = camMatActive;
    selectedDepthCamera = camObj.camera;
    
    computePeelingLayers();
    
    const origin = camObj.camera.position;
    const dir = new THREE.Vector3(0,0,-1).applyQuaternion(camObj.camera.quaternion);
    const localRaycaster = new THREE.Raycaster(origin, dir);
    currentIntersects = localRaycaster.intersectObject(torus);
    currentIntersects.sort((a, b) => a.distance - b.distance);
    
    hitSpheresGroup.clear();
    hitSpheres = [];
    const hitColors = [0x10b981, 0x8b5cf6, 0x10b981, 0x8b5cf6]; 
    currentIntersects.forEach((hit, idx) => {
        const geo = new THREE.SphereGeometry(0.8, 16, 16);
        const mat = new THREE.MeshBasicMaterial({ color: hitColors[idx % 4] });
        const sphere = new THREE.Mesh(geo, mat);
        sphere.position.copy(hit.point);
        hitSpheresGroup.add(sphere);
        hitSpheres.push(sphere);
    });
    
    resetTriangleSelection();
    flyToCamera(camObj.camera);
    
    // Only update UI if we are in Step 0
    if (currentStep === 0) updateUI();
}

function flyToCamera(targetCam) {
    controlsEnabled = false;
    new TWEEN.Tween(camera.position)
        .to({ x: targetCam.position.x, y: targetCam.position.y, z: targetCam.position.z }, 1000)
        .easing(TWEEN.Easing.Exponential.InOut)
        .onUpdate(() => { controls.target.set(0,0,0); controls.update(); })
        .onComplete(() => { controlsEnabled = true; })
        .start();
}

function resetTriangleSelection() {
    torusMaterial.opacity = 0.35;
    if (activeTriangleMesh) {
        scene.remove(activeTriangleMesh);
        activeTriangleMesh = null;
        activeTriangleGeo = null;
    }
}

function selectTriangle(hit) {
    if (hit.faceIndex === undefined) return;
    
    resetTriangleSelection();
    torusMaterial.opacity = 0.05; 
    
    const faceIndex = hit.faceIndex;
    const geom = torusGeometry;
    const positions = geom.attributes.position;
    const indices = geom.index;
    
    const triGeo = new THREE.BufferGeometry();
    const newPos = new Float32Array(9); 
    let centroid = new THREE.Vector3();
    
    for(let i=0; i<3; i++) {
        let vId = indices.getX(faceIndex * 3 + i);
        let x = positions.getX(vId);
        let y = positions.getY(vId);
        let z = positions.getZ(vId);
        newPos[i*3] = x;
        newPos[i*3+1] = y;
        newPos[i*3+2] = z;
        centroid.add(new THREE.Vector3(x, y, z));
    }
    centroid.divideScalar(3);
    centroid.applyMatrix4(torus.matrixWorld);
    
    triGeo.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
    
    const wireMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 3 });
    const wireGeo = new THREE.WireframeGeometry(triGeo);
    activeTriangleMesh = new THREE.LineSegments(wireGeo, wireMat);
    activeTriangleMesh.rotation.copy(torus.rotation);
    activeTriangleMesh.position.copy(torus.position);
    scene.add(activeTriangleMesh);
    
    activeTriangleGeo = triGeo;
    zoomToTriangle(centroid);
}

function autoSelectRandomTriangle() {
    if (!activeCameraObj || currentIntersects.length === 0) return;
    selectTriangle(currentIntersects[0]);
}

function zoomToTriangle(centroid) {
    controlsEnabled = false;
    const dir = new THREE.Vector3().subVectors(centroid, camera.position).normalize();
    const dist = camera.position.distanceTo(centroid);
    const targetDist = 5; 
    const zoomVector = dir.clone().multiplyScalar(dist - targetDist);
    const newPos = camera.position.clone().add(zoomVector);
    
    new TWEEN.Tween(controls.target)
        .to({ x: centroid.x, y: centroid.y, z: centroid.z }, 1000)
        .easing(TWEEN.Easing.Cubic.InOut)
        .start();
        
    new TWEEN.Tween(camera.position)
        .to({ x: newPos.x, y: newPos.y, z: newPos.z }, 1000)
        .easing(TWEEN.Easing.Cubic.InOut)
        .onUpdate(() => { controls.update(); })
        .onComplete(() => { controlsEnabled = true; })
        .start();
}

renderer.domElement.addEventListener('pointerdown', (event) => {
    // Only allow selecting during step 0
    if (currentStep !== 0) return;
    
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    
    const meshes = cameraObjects.map(obj => obj.mesh);
    const camIntersects = raycaster.intersectObjects(meshes);
    
    if (camIntersects.length > 0) {
        const clickedMesh = camIntersects[0].object;
        selectCamera(clickedMesh.userData);
        return;
    }
    
    if (activeCameraObj) {
        const torusIntersects = raycaster.intersectObject(torus);
        if (torusIntersects.length > 0) {
            selectTriangle(torusIntersects[0]);
        }
    }
});

// --- RASTERIZE ANIMATION VARIABLES ---
const rasterGroup = new THREE.Group();
scene.add(rasterGroup);
let rasterTweens = [];

function clearRasterAnimation() {
    rasterTweens.forEach(t => t.stop());
    rasterTweens = [];
    rasterGroup.clear();
    
    // Restore UI defaults
    torus.visible = true;
    if (activeTriangleMesh) activeTriangleMesh.visible = true;
    cameraObjects.forEach(obj => { obj.helper.visible = true; obj.mesh.visible = true; });
    hitSpheresGroup.visible = true;
}

function getTriangleVerticesAndColors() {
    const posAttr = activeTriangleGeo.attributes.position;
    const v1 = new THREE.Vector3(posAttr.getX(0), posAttr.getY(0), posAttr.getZ(0)).applyMatrix4(torus.matrixWorld);
    const v2 = new THREE.Vector3(posAttr.getX(1), posAttr.getY(1), posAttr.getZ(1)).applyMatrix4(torus.matrixWorld);
    const v3 = new THREE.Vector3(posAttr.getX(2), posAttr.getY(2), posAttr.getZ(2)).applyMatrix4(torus.matrixWorld);
    
    const calcColor = (v) => {
        const dist = selectedDepthCamera.position.distanceTo(v);
        // Camera is at distance 20. Torus radius is ~11. 
        // Closest point is ~9. Torus center is ~20.
        // Map distance 9 to 1.0 (White), and distance 20 to 0.0 (Black)
        let intensity = 1.0 - ((dist - 9) / 11);
        intensity = Math.max(0, Math.min(1, intensity));
        return new THREE.Color(intensity, intensity, intensity);
    };
    return { v1, v2, v3, c1: calcColor(v1), c2: calcColor(v2), c3: calcColor(v3) };
}

function playRasterizeStep1() {
    clearRasterAnimation();
    if (!activeTriangleGeo) autoSelectRandomTriangle();
    if (!activeTriangleGeo) return; // Fallback failed
    torus.visible = true;
    torus.material.opacity = 0.05;
    activeTriangleMesh.visible = false;
    cameraObjects.forEach(obj => { obj.helper.visible = false; obj.mesh.visible = false; });
    hitSpheresGroup.visible = false;
    
    const {v1, v2, v3, c1, c2, c3} = getTriangleVerticesAndColors();
    
    const createSphere = (p, c) => {
        const mesh = new THREE.Mesh(new THREE.SphereGeometry(1.0, 16, 16), new THREE.MeshBasicMaterial({color: c}));
        mesh.position.copy(p);
        mesh.scale.set(0,0,0);
        mesh.userData.isVertexSphere = true;
        rasterGroup.add(mesh);
        return mesh;
    };
    
    const s1 = createSphere(v1, c1);
    const s2 = createSphere(v2, c2);
    
    // Draw line from V1 to V2
    const lineGeo = new THREE.BufferGeometry();
    const linePos = new Float32Array([v1.x, v1.y, v1.z, v1.x, v1.y, v1.z]);
    const lineCol = new Float32Array([c1.r, c1.g, c1.b, c1.r, c1.g, c1.b]);
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
    lineGeo.setAttribute('color', new THREE.BufferAttribute(lineCol, 3));
    const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 4 });
    const lineMesh = new THREE.Line(lineGeo, lineMat);
    rasterGroup.add(lineMesh);
    
    // Animate Spheres and Line simultaneously
    s1.scale.set(pointSize, pointSize, pointSize); 
    s2.scale.set(pointSize, pointSize, pointSize);
    const tw = new TWEEN.Tween({t: 0}).to({t: 1}, 2000).easing(TWEEN.Easing.Quadratic.InOut)
        .onUpdate(obj => {
            const currentPos = v1.clone().lerp(v2, obj.t);
            const currentCol = c1.clone().lerp(c2, obj.t);
            linePos[3] = currentPos.x; linePos[4] = currentPos.y; linePos[5] = currentPos.z;
            lineCol[3] = currentCol.r; lineCol[4] = currentCol.g; lineCol[5] = currentCol.b;
            lineGeo.attributes.position.needsUpdate = true;
            lineGeo.attributes.color.needsUpdate = true;
        }).start();
    rasterTweens.push(tw);
}

function playRasterizeStep2() {
    clearRasterAnimation();
    if (!activeTriangleGeo) return;
    torus.visible = true;
    torus.material.opacity = 0.05;
    activeTriangleMesh.visible = false;
    cameraObjects.forEach(obj => { obj.helper.visible = false; obj.mesh.visible = false; });
    hitSpheresGroup.visible = false;
    
    const {v1, v2, v3, c1, c2, c3} = getTriangleVerticesAndColors();
    
    const s3 = new THREE.Mesh(new THREE.SphereGeometry(1.0, 16, 16), new THREE.MeshBasicMaterial({color: c3}));
    s3.position.copy(v3);
    s3.scale.set(pointSize, pointSize, pointSize);
    s3.userData.isVertexSphere = true;
    rasterGroup.add(s3);
    
    // Full line already drawn
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([v1.x, v1.y, v1.z, v2.x, v2.y, v2.z]), 3));
    lineGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array([c1.r, c1.g, c1.b, c2.r, c2.g, c2.b]), 3));
    rasterGroup.add(new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 4 })));
    
    // Face Fill Quad
    const faceGeo = new THREE.BufferGeometry();
    const facePos = new Float32Array(18);
    const faceCol = new Float32Array(18);
    const setVert = (idx, v, c) => {
        facePos[idx*3]=v.x; facePos[idx*3+1]=v.y; facePos[idx*3+2]=v.z;
        faceCol[idx*3]=c.r; faceCol[idx*3+1]=c.g; faceCol[idx*3+2]=c.b;
    };
    faceGeo.setAttribute('position', new THREE.BufferAttribute(facePos, 3));
    faceGeo.setAttribute('color', new THREE.BufferAttribute(faceCol, 3));
    const faceMat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    rasterGroup.add(new THREE.Mesh(faceGeo, faceMat));
    
    const tw = new TWEEN.Tween({t: 0}).to({t: 1}, 3000).easing(TWEEN.Easing.Cubic.InOut)
        .onUpdate(obj => {
            const L_left = v1.clone().lerp(v3, obj.t);
            const c_left = c1.clone().lerp(c3, obj.t);
            const L_right = v2.clone().lerp(v3, obj.t);
            const c_right = c2.clone().lerp(c3, obj.t);
            
            setVert(0, v1, c1); setVert(1, v2, c2); setVert(2, L_right, c_right);
            setVert(3, v1, c1); setVert(4, L_right, c_right); setVert(5, L_left, c_left);
            
            faceGeo.attributes.position.needsUpdate = true;
            faceGeo.attributes.color.needsUpdate = true;
        }).start();
    rasterTweens.push(tw);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate(time) {
    requestAnimationFrame(animate);
    TWEEN.update(time);
    if (controlsEnabled) controls.update();
    renderer.render(scene, camera);
}
animate();

function renderDepth(material, clippingPlanes, statusText) {
    if (!torus.visible) {
        depthRenderer.clear();
        monitorStatus.textContent = "Offline";
        return;
    }
    torusDepth.material = material;
    torusDepth.material.clippingPlanes = clippingPlanes;
    
    const depthScene = new THREE.Scene();
    depthScene.background = new THREE.Color(0x000000);
    depthScene.add(torusDepth);
    depthRenderer.render(depthScene, selectedDepthCamera);
    monitorStatus.textContent = statusText;
}

// --- STATE MACHINE ---
let currentStep = 0;
let useOutliers = false;
let numRays = 1;
let pointSize = 0.05;

const steps = [
    {
        title: "Step 1: Select a Camera & Triangle",
        text: "You can freely click on any yellow camera cone, and then click any Triangle on the blue Torus to isolate it.",
        apply: () => {
            clearRasterAnimation();
            applyPeelLevel(0);
            torus.material = torusMaterial;
            hitSpheres.forEach(s => s.visible = false);
            renderDepth(depthMaterial, [], "Offline");
            monitorStatus.style.color = "#94a3b8";
        }
    },
    {
        title: "Step 2: Rasterize Edge Interpolation",
        text: "The GPU extracts 2 vertices from the 3D model and interpolates their depth (color) to draw a perfectly blended line (Edge).",
        apply: () => {
            playRasterizeStep1();
            torus.material = torusMaterial;
            renderDepth(depthMaterial, [], "Offline");
            monitorStatus.style.color = "#94a3b8";
        }
    },
    {
        title: "Step 3: Rasterize Face Sweep",
        text: "Using Barycentric coordinates, the GPU scans from that line towards the 3rd vertex, filling the entire triangle with interpolated depth values.",
        apply: () => {
            playRasterizeStep2();
            torus.material = torusMaterial;
            renderDepth(depthMaterial, [], "Offline");
            monitorStatus.style.color = "#94a3b8";
        }
    },
    {
        title: "Step 4: Rasterize Full Object",
        text: "The GPU repeats this Barycentric interpolation process for all triangles simultaneously. The result is a complete Depth Map for the entire object!",
        apply: () => {
            clearRasterAnimation();
            applyPeelLevel(0);
            renderDepth(depthMaterial, [], "Rasterized Full Object");
            
            customDepthMaterial.uniforms.camPos.value.copy(selectedDepthCamera.position);
            customDepthMaterial.side = THREE.DoubleSide;
            customDepthMaterial.transparent = false;
            customDepthMaterial.opacity = 1.0;
            customDepthMaterial.depthWrite = true;
            torus.material = customDepthMaterial;
            torus.material.clippingPlanes = [];
            hitSpheres.forEach((s) => s.visible = false);
            monitorStatus.style.color = "#10b981";
        }
    },
    {
        title: "Step 5: Peel Iteration 1 (Front)",
        text: "First Depth Peeling pass. The GPU captures Layer 1, then we literally DELETE those triangles from the 3D view to reveal what's underneath!",
        apply: () => {
            clearRasterAnimation();
            
            // 1. Capture Layer 1 before deleting
            applyPeelLevel(0);
            renderDepth(depthMaterial, [], "Captured: depthTextureFront");
            
            // 2. Delete Layer 1
            applyPeelLevel(1);
            
            customDepthMaterial.uniforms.camPos.value.copy(selectedDepthCamera.position);
            customDepthMaterial.side = THREE.DoubleSide;
            customDepthMaterial.transparent = false;
            customDepthMaterial.opacity = 1.0;
            customDepthMaterial.depthWrite = true;
            torus.material = customDepthMaterial;
            torus.material.clippingPlanes = [];
            hitSpheres.forEach((s, i) => s.visible = (i === 1)); // Show Layer 2 hitting
            monitorStatus.style.color = "#10b981";
        }
    },
    {
        title: "Step 6: Peel Iteration 2 (Back)",
        text: "Second peeling pass captures Layer 2, then we DELETE Layer 2 triangles. Now we can see straight through the front tube!",
        apply: () => {
            clearRasterAnimation();
            
            // 1. Capture Layer 2
            applyPeelLevel(1);
            renderDepth(depthMaterial, [], "Captured: depthTextureBack");
            
            // 2. Delete Layer 2
            applyPeelLevel(2);
            
            customDepthMaterial.uniforms.camPos.value.copy(selectedDepthCamera.position);
            customDepthMaterial.side = THREE.DoubleSide;
            customDepthMaterial.transparent = false;
            customDepthMaterial.opacity = 1.0;
            customDepthMaterial.depthWrite = true;
            torus.material = customDepthMaterial;
            torus.material.clippingPlanes = [];
            hitSpheres.forEach((s, i) => s.visible = (i === 1 || i === 2));
            monitorStatus.style.color = "#8b5cf6";
        }
    },
    {
        title: "Step 7: Calculate SDF",
        text: "SDF is the thickness between the Front and Back hits. SDF = Back Depth - Front Depth.",
        apply: () => {
            clearRasterAnimation();
            applyPeelLevel(2);
            renderDepth(depthMaterial, [], "Calculating SDF (Shader)");
            
            customDepthMaterial.uniforms.camPos.value.copy(selectedDepthCamera.position);
            customDepthMaterial.side = THREE.DoubleSide;
            customDepthMaterial.transparent = false;
            customDepthMaterial.opacity = 1.0;
            customDepthMaterial.depthWrite = true;
            torus.material = customDepthMaterial;
            torus.material.clippingPlanes = [];
            hitSpheres.forEach((s, i) => s.visible = (i === 0 || i === 1));
            monitorStatus.style.color = "#0ea5e9";
        }
    }
];

function getSteps() {
    let activeSteps = [...steps];
    if (useOutliers) {
        activeSteps.push({
            title: "Step 7.1: Outlier Removal (Median)",
            text: "The Fragment Shader samples 81 neighboring pixels (9x9 grid) around the intersection on the depth buffer and takes the MEDIAN. Extremely heavy on GPU.",
            apply: () => {
                clearRasterAnimation();
                applyPeelLevel(2);
                renderDepth(depthMaterial, [], "Sampling 9x9 pixels...");
                
                customDepthMaterial.uniforms.camPos.value.copy(selectedDepthCamera.position);
                customDepthMaterial.side = THREE.DoubleSide;
                customDepthMaterial.transparent = false;
                customDepthMaterial.opacity = 1.0;
                customDepthMaterial.depthWrite = true;
                torus.material = customDepthMaterial;
                torus.material.clippingPlanes = [];
                hitSpheres.forEach((s, i) => s.visible = (i === 0 || i === 1));
                monitorStatus.style.color = "#f59e0b";
            }
        });
    }
    
    activeSteps.push({
        title: "Step 8: Peel Iteration 3 (Front Inner)",
        text: "Wait! A Torus has a hole. The ray continues and hits the Front Inner face on the other side. So we DELETE Layer 3 triangles too!",
        apply: () => {
            clearRasterAnimation();
            
            // 1. Capture Layer 3
            applyPeelLevel(2);
            renderDepth(depthMaterial, [], "Captured: Peel Layer 3");
            
            // 2. Delete Layer 3
            applyPeelLevel(3);
            
            customDepthMaterial.uniforms.camPos.value.copy(selectedDepthCamera.position);
            customDepthMaterial.side = THREE.DoubleSide;
            customDepthMaterial.transparent = false;
            customDepthMaterial.opacity = 1.0;
            customDepthMaterial.depthWrite = true;
            torus.material = customDepthMaterial;
            torus.material.clippingPlanes = [];
            hitSpheres.forEach((s, i) => s.visible = (i === 3)); // Show Layer 4 hitting
            monitorStatus.style.color = "#10b981";
        }
    });
    
    activeSteps.push({
        title: "Step 9: Peel Iteration 4 (Back Outer)",
        text: "Finally, it hits Layer 4. We DELETE Layer 4 triangles. Now we have a clean tunnel carved straight through the entire Torus!",
        apply: () => {
            clearRasterAnimation();
            
            // 1. Capture Layer 4
            applyPeelLevel(3);
            renderDepth(depthMaterial, [], "Captured: Peel Layer 4");
            
            // 2. Delete Layer 4
            applyPeelLevel(4);
            
            customDepthMaterial.uniforms.camPos.value.copy(selectedDepthCamera.position);
            customDepthMaterial.side = THREE.DoubleSide;
            customDepthMaterial.transparent = false;
            customDepthMaterial.opacity = 1.0;
            customDepthMaterial.depthWrite = true;
            torus.material = customDepthMaterial;
            torus.material.clippingPlanes = [];
            hitSpheres.forEach((s, i) => s.visible = false); // All deleted
            monitorStatus.style.color = "#8b5cf6";
        }
    });
    
    return activeSteps;
}

function updateUI() {
    const activeSteps = getSteps();
    const stepData = activeSteps[currentStep];

    statusOverlay.textContent = stepData.title;
    statusOverlay.style.color = "white";
    explanationText.textContent = stepData.text;

    if (stepData.title.includes("Select") || stepData.title.includes("Rasterize")) {
        valFront.textContent = "-"; valBack.textContent = "-"; valSDF.textContent = "-";
    } else if (stepData.title.includes("Iteration 1")) {
        valFront.textContent = currentIntersects.length > 0 ? currentIntersects[0].distance.toFixed(2) : "-"; 
        valBack.textContent = "-"; valSDF.textContent = "-";
    } else if (stepData.title.includes("Iteration 2") || stepData.title.includes("Calculate SDF") || stepData.title.includes("Outlier")) {
        valFront.textContent = currentIntersects.length > 0 ? currentIntersects[0].distance.toFixed(2) : "-"; 
        valBack.textContent = currentIntersects.length > 1 ? currentIntersects[1].distance.toFixed(2) : "-"; 
        valSDF.textContent = currentIntersects.length > 1 ? (currentIntersects[1].distance - currentIntersects[0].distance).toFixed(2) : "-";
    }

    btnPrev.disabled = currentStep === 0;
    btnNext.textContent = currentStep === activeSteps.length - 1 ? "Finish" : "Next Step";
    btnNext.disabled = currentStep === activeSteps.length - 1;

    stepData.apply();
}

btnNext.addEventListener('click', () => {
    if (currentStep < getSteps().length - 1) {
        currentStep++;
        updateUI();
    }
});

btnPrev.addEventListener('click', () => {
    if (currentStep > 0) {
        currentStep--;
        updateUI();
    }
});

toggleOutliers.addEventListener('change', (e) => {
    useOutliers = e.target.checked;
    currentStep = 0; 
    updateUI();
});

numRaysSlider.addEventListener('input', (e) => {
    numRays = parseInt(e.target.value);
    numRaysDisplay.textContent = numRays;
    updateMultiRays(numRays);
    updateUI();
});

pointSizeSlider.addEventListener('input', (e) => {
    pointSize = parseFloat(e.target.value);
    pointSizeDisplay.textContent = pointSize.toFixed(2);
    
    // Instantly update active spheres if they exist
    rasterGroup.children.forEach(child => {
        if (child.userData.isVertexSphere) {
            child.scale.set(pointSize, pointSize, pointSize);
        }
    });
});

updateMultiRays(1); 
updateUI();
