import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import TWEEN from '@tweenjs/tween.js';

import { setupGeometry } from './modules/geometry.js?v=2';
import { setupStepManager } from './modules/stepManager.js?v=2';

window.TWEEN = TWEEN;
console.log("[Init] Starting main.js...");

try {
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 5, 15);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('canvas-container').appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Add Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

// Shared Context
const SC = {
    scene, camera, renderer, controls,
    isAnimating: false,
    currentStep: 1,
    tweenGroup: new TWEEN.Group()
};
window.SC = SC;

// Initialize modules
const geoData = setupGeometry(SC);
Object.assign(SC, geoData);

const stepManager = setupStepManager(SC);
Object.assign(SC, stepManager);

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Keyboard and Button controls
document.addEventListener('keydown', (e) => {
    console.log(`[Key Pressed] ${e.key}`);
    if (e.key === 'ArrowRight') {
        SC.goToStep(SC.currentStep + 1);
    } else if (e.key === 'ArrowLeft') {
        SC.goToStep(SC.currentStep - 1);
    }
});

const btnNext = document.getElementById('btn-next');
if (btnNext) {
    console.log("[Init] Next Step button found.");
    btnNext.addEventListener('click', (e) => {
        e.stopPropagation();
        console.log(`[Button Clicked] Next Step. Current Step is ${SC.currentStep}`);
        SC.goToStep(SC.currentStep + 1);
    });
} else {
    console.warn("[Init] btn-next NOT FOUND in DOM!");
}

const btnPrev = document.getElementById('btn-prev');
if (btnPrev) {
    console.log("[Init] Prev Step button found.");
    btnPrev.addEventListener('click', (e) => {
        e.stopPropagation();
        console.log(`[Button Clicked] Prev Step. Current Step is ${SC.currentStep}`);
        SC.goToStep(SC.currentStep - 1);
    });
} else {
    console.warn("[Init] btn-prev NOT FOUND in DOM!");
}

function animate(time) {
    requestAnimationFrame(animate);
    TWEEN.update(time);
    controls.update();
    renderer.render(scene, camera);
}
animate(performance.now());

// Start at step 1
setTimeout(() => SC.goToStep(1), 100);

} catch (e) {
    console.error("[Fatal Error]", e);
    alert("Lỗi xảy ra trong main.js: " + e.message);
}
