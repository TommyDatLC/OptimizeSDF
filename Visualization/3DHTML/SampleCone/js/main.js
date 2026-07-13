
import * as THREE from 'three';
import TWEEN from '@tweenjs/tween.js';
window.TWEEN = TWEEN;
import { setupScene } from './modules/sceneSetup.js?v=41';
import { setupGeometry } from './modules/geometry.js?v=41';
import { setupRays } from './modules/rayManager.js?v=41';
import { setupStepManager } from './modules/stepManager.js?v=41';

const ptData = [];
const ptMeshes = [];
const radialLines = [];
window.ptData = ptData;
window.ptMeshes = ptMeshes;
window.radialLines = radialLines;

const SC = {};

const sceneVars = setupScene();
Object.assign(SC, sceneVars);

const geomVars = setupGeometry(SC);
Object.assign(SC, geomVars);

const rayVars = setupRays(SC, ptData, ptMeshes, radialLines);
Object.assign(SC, rayVars);

// Initialize step manager
const stepVars = setupStepManager(SC);
Object.assign(SC, stepVars);

// Expose globally
window.SC = SC;
window.currentStep = 0;

// Setup pointer listeners
let startX, startY;
let lastClickTime = 0;
document.addEventListener('contextmenu', e => e.preventDefault());
document.addEventListener('pointerdown', e => { startX = e.clientX; startY = e.clientY; });
document.addEventListener('pointerup', e => {
    const now = performance.now();
    if (now - lastClickTime < 100) return; // Prevent duplicate rapid events
    lastClickTime = now;
    
    const dx = e.clientX - startX, dy = e.clientY - startY;
    if (Math.sqrt(dx*dx + dy*dy) < 5) {
        if (e.button === 0) SC.goToStep(window.currentStep + 1);
        else if (e.button === 2) SC.goToStep(window.currentStep - 1);
    }
});

window.addEventListener('resize', () => {
    SC.camera.aspect = window.innerWidth / window.innerHeight;
    SC.camera.updateProjectionMatrix();
    SC.renderer.setSize(window.innerWidth, window.innerHeight);
});

SC.snapToStep(1);

function animate(time) {
        requestAnimationFrame(animate);
        if(window.tweenGroup) window.tweenGroup.update(time);

        // Liên tục cập nhật các đường tia đỏ bám dính vào hạt


        radialLines.forEach((rl, i) => {
            if (rl.visible) {
                const pt = ptMeshes[i].position;
                let endPt = pt;
                if (window.currentStep === 3 && ptData[i] && !ptData[i].isUser) {
                    endPt = pt.clone().normalize().multiplyScalar(50);
                }
                rl.geometry.setFromPoints([new THREE.Vector3(0, 0, 0), endPt]);
            }
        });

        // Auto Rotate chỉ hoạt động khi Camera không bị Panning bởi Code
        SC.controls.autoRotate = !window.isCameraPanning;
        SC.controls.update();

        SC.updateLabels();
        SC.renderer.render(SC.scene, SC.camera);
    }
animate();
