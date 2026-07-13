import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function setupScene() {
    // 1. SETUP SCENE, CAMERA, RENDERER
    // ==========================================
    const scene = new THREE.Scene();
    const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    gridHelper.visible = false;
    scene.add(gridHelper);

    // Grid toggle
    let gridVisible = false;
    const gridToggle = document.getElementById('grid-toggle');
    gridToggle.addEventListener('click', () => {
        gridVisible = !gridVisible;
        gridHelper.visible = gridVisible;
        gridToggle.textContent = gridVisible ? '🔲 Grid: ON' : '🔲 Grid: OFF';
    });

    // Background toggle
    let isDark = true;
    const bgToggle = document.getElementById('bg-toggle');
    bgToggle.addEventListener('click', () => {
        isDark = !isDark;
        if (isDark) {
            document.body.style.backgroundColor = '#080808';
            scene.background = null;
            gridHelper.material[0].color.setHex(0x444444);
            gridHelper.material[1].color.setHex(0x222222);
            bgToggle.textContent = '🌙 Dark';
        } else {
            document.body.style.backgroundColor = '#ffffff';
            scene.background = new THREE.Color(0xffffff);
            gridHelper.material[0].color.setHex(0xcccccc);
            gridHelper.material[1].color.setHex(0xdddddd);
            bgToggle.textContent = '☀️ Light';
        }
    });

    const coneInitialPos = new THREE.Vector3(0, 0, 0);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    const startCamPos = new THREE.Vector3(0, 4, 8);
    camera.position.copy(startCamPos);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    // Bật tính năng tự động xoay camera
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.0;
    controls.target.set(0, 1.5, 0);
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: null };

    window.isCameraPanning = false;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    // ==========================================
    
    return { scene, camera, renderer, controls, startCamPos, coneInitialPos, gridHelper };
}
