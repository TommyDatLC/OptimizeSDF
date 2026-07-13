import * as THREE from 'three';
import TWEEN from '@tweenjs/tween.js';

export function setupRays(SC, ptData, ptMeshes, radialLines) {
    const { scene, coneGroup, targetVertex, camera, coneHeight, coneRadius, mathGroup } = SC;
    // 4. RANDOM POINTS (UNIT EDGE -> EDGE -> INNER)
    // ==========================================
    
    const coneHalfAngle = Math.atan(SC.coneRadius / SC.coneHeight);

function updatePointPosition(idx) {
        const data = ptData[idx];
        const R = SC.coneHeight / Math.cos(coneHalfAngle);
        const cosThetaMax = Math.cos(coneHalfAngle);
        
        const y_inner = R * (cosThetaMax + (1.0 - cosThetaMax) * data.uvx);
        const r_inner = Math.sqrt(Math.max(0, R * R - y_inner * y_inner));
        const phi = 2.0 * Math.PI * data.uvy;
        
        data.pInner = new THREE.Vector3(r_inner * Math.cos(phi), y_inner, r_inner * Math.sin(phi));
        
        
        if (ptMeshes[idx]) ptMeshes[idx].position.copy(data.pInner);
        if (radialLines[idx]) radialLines[idx].geometry.setFromPoints([new THREE.Vector3(0,0,0), data.pInner]);

        

        
        updateRayIntersections();
    }

    const pointsGroup = new THREE.Group();
    coneGroup.add(pointsGroup);
    const numPoints = 12;

    
    
    

    const circlePts = [];
    for(let i=0; i<=64; i++) {
        const a = i/64*Math.PI*2;
        circlePts.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a)));
    }
    const circleGeo = new THREE.BufferGeometry().setFromPoints(circlePts);
    const circleLine = new THREE.Line(circleGeo, new THREE.LineBasicMaterial({color: 0xAAAAAA, transparent: true, opacity: 0}));
    circleLine.position.set(0, SC.coneHeight, 0);
    circleLine.visible = false;
    coneGroup.add(circleLine);

    const centerBase = new THREE.Vector3(0, SC.coneHeight, 0);

    for(let i=0; i<numPoints; i++){
        const theta_rot = Math.random() * Math.PI * 2;
        const u1 = Math.random();

        const pUnitEdge = new THREE.Vector3(Math.cos(theta_rot), SC.coneHeight, Math.sin(theta_rot));
        const pEdge = new THREE.Vector3(SC.coneRadius * Math.cos(theta_rot), SC.coneHeight, SC.coneRadius * Math.sin(theta_rot));

        const r_inner = SC.coneRadius * Math.sqrt(u1);
        const pInner = new THREE.Vector3(r_inner * Math.cos(theta_rot), SC.coneHeight, r_inner * Math.sin(theta_rot));

        ptData.push({ pUnitEdge, pEdge, pInner });

        const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.08, 8, 8), new THREE.MeshBasicMaterial({ color: 0xFFFFFF }));
        mesh.position.copy(pUnitEdge);
        mesh.visible = false;
        pointsGroup.add(mesh);
        ptMeshes.push(mesh);

        // Đường bán kính liên kết động với các hạt (Màu Đỏ)
        const rGeo = new THREE.BufferGeometry().setFromPoints([centerBase, pUnitEdge]);
        const rMat = new THREE.LineBasicMaterial({ color: 0xFF0000, transparent: true, opacity: 0, linewidth: 2 });
        const rLineObj = new THREE.Line(rGeo, rMat);
        rLineObj.visible = false;
        coneGroup.add(rLineObj);
        radialLines.push(rLineObj);
    }

    // ==========================================

    // 6. TIA RAY VÀ TÍNH TOÁN RAYCAST



    const rays = [];
    const intersectDots = [];
    const rayMatObj = new THREE.LineBasicMaterial({ color: 0xFFFF00, transparent: true, opacity: 0, linewidth: 2 });
    for(let i=0; i<12; i++) {
        const geo = new THREE.BufferGeometry().setFromPoints([targetVertex, targetVertex]);
        const line = new THREE.Line(geo, rayMatObj);
        line.visible = false;
        scene.add(line);
        rays.push(line);
        
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 8), new THREE.MeshBasicMaterial({ color: 0xFF0000 }));
        dot.visible = false;
        scene.add(dot);
        intersectDots.push(dot);
    }

    const raycaster = new THREE.Raycaster();
    const intersectPts = [];
    const hitDistances = [];

    const finalQuat = SC.finalQuat;


function addRayData(uvx, uvy, isUser = false) {
        const R = SC.coneHeight / Math.cos(coneHalfAngle);
        const cosThetaMax = Math.cos(coneHalfAngle);
        const y_inner = R * (cosThetaMax + (1.0 - cosThetaMax) * uvx);
        const r_inner = Math.sqrt(Math.max(0, R * R - y_inner * y_inner));
        const phi = 2.0 * Math.PI * uvy;
        const pInner = new THREE.Vector3(r_inner * Math.cos(phi), y_inner, r_inner * Math.sin(phi));
        
        const idx = ptData.length;
        
        ptData.push({ uvx, uvy, pInner, isUser });
        
        const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.08, 8, 8), new THREE.MeshBasicMaterial({ color: isUser ? 0x00FF00 : 0xFFFFFF }));
        mesh.position.copy(pInner);
        mesh.visible = window.currentStep >= 1 && window.currentStep < 13;
        pointsGroup.add(mesh);
        ptMeshes.push(mesh);
        
        const rGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), pInner]);
        const rLineObj = new THREE.Line(rGeo, new THREE.LineBasicMaterial({ color: 0xFF0000, transparent: true, opacity: window.currentStep >= 1 && window.currentStep < 13 ? 0.8 : 0, linewidth: 2 }));
        rLineObj.visible = window.currentStep >= 1 && window.currentStep < 12;
        coneGroup.add(rLineObj);
        radialLines.push(rLineObj);
        
        if (targetVertex) {
            const rayLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([targetVertex, targetVertex]), rayMatObj.clone());
            rayLine.visible = window.currentStep >= 12;
            scene.add(rayLine);
            rays.push(rayLine);
            
            const dot = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 8), new THREE.MeshBasicMaterial({ color: 0xFF0000 }));
            dot.visible = window.currentStep >= 13;
            scene.add(dot);
            intersectDots.push(dot);
        }
        
        updatePointPosition(idx);
        
        if (isUser) {
            const list1 = document.getElementById('ray-list-step1');
            const d1 = document.createElement('div');
            d1.className = 'ray-item';
            d1.innerHTML = `Ray #${idx+1}: <span style="font-family: math; font-size: 1.1em;">y = R [ cos(&theta;<sub>max</sub>) + (1 - cos(&theta;<sub>max</sub>)) &times; <input type="number" step="0.01" min="0" max="1" value="${uvx.toFixed(2)}" data-idx="${idx}"> ]</span>`;
            if (list1) list1.appendChild(d1);
            
            const list2 = document.getElementById('ray-list-step2');
            const d2 = document.createElement('div');
            d2.className = 'ray-item';
            d2.innerHTML = `Ray #${idx+1}: <span style="font-family: math; font-size: 1.1em;">&phi; = 2&pi; &times; <input type="number" step="0.01" min="0" max="1" value="${uvy.toFixed(2)}" data-idx="${idx}"></span>`;
            if (list2) list2.appendChild(d2);
        }
return idx;
    }



function updateRayIntersections() {
        if (!targetVertex || ptData.length === 0) return;
        intersectPts.length = 0;
        hitDistances.length = 0;
        for(let i=0; i<ptData.length; i++){
            if (i >= rays.length) {
                const rayLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([targetVertex, targetVertex]), rayMatObj.clone());
                rayLine.visible = window.currentStep >= 12;
                scene.add(rayLine);
                rays.push(rayLine);
                
                const dot = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 8), new THREE.MeshBasicMaterial({ color: 0xFF0000 }));
                dot.visible = window.currentStep >= 13;
                scene.add(dot);
                intersectDots.push(dot);
            }
            
            const localDir = ptData[i].pInner.clone().normalize();
            const worldDir = localDir.clone().applyQuaternion(finalQuat).normalize();
            raycaster.set(targetVertex.clone().add(worldDir.clone().multiplyScalar(0.01)), worldDir);
            const intersects = raycaster.intersectObjects(SC.faces, false);
            if(intersects.length > 0) {
                intersectPts.push(intersects[0].point);
                hitDistances.push(intersects[0].distance + 0.01);
            } else {
                intersectPts.push(targetVertex.clone().add(worldDir.clone().multiplyScalar(15)));
                hitDistances.push(15);
            }
        }
        const validHits = hitDistances.filter(d => d < 15);
        SC.avgDist = ptData.length > 0 ? (validHits.reduce((a,b)=>a+b, 0) / ptData.length) : 0;
        
        if (window.currentStep === 13) {
            SC.mathFormulas[13] = `\\text{Step 13:} \\quad d_{avg} = \\frac{1}{${ptData.length}} \\sum d_{i\\_rand} \\approx ${SC.avgDist.toFixed(2)}`;
            const mathOverlay = document.getElementById('math-formula');
            if (mathOverlay && mathOverlay.style.display !== 'none') {
                katex.render(SC.mathFormulas[13], mathOverlay, { throwOnError: false });
            }
        }
    }

    function getTurboColor(x) {
        x = Math.max(0.0, Math.min(1.0, x));
        const x2 = x*x; const x3 = x2*x; const x4 = x3*x; const x5 = x4*x;
        const r = 0.13572138 + 4.61539260*x - 42.66032258*x2 + 132.12590289*x3 - 152.94239396*x4 + 59.05650193*x5;
        const g = 0.09140261 + 2.19418839*x + 4.84296658*x2 - 14.18503333*x3 + 4.27729857*x4 + 2.82956604*x5;
        const b = 0.10667330 + 12.64194608*x - 60.58204836*x2 + 110.36276771*x3 - 89.90310912*x4 + 27.34824973*x5;
        return new THREE.Color(Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b)));
    }


let globalVertexSDFs = null;
let globalMinSDF = Infinity;
let globalMaxSDF = -Infinity;
let activeVertices = [];

function resetHeatmap() {
    activeVertices = [];
    const c0 = getHeatmapColor(0);
    SC.faces.forEach(mesh => {
        const geo = mesh.geometry;
        let colAttr = geo.attributes.color;
        if (!colAttr) {
            const posAttr = geo.attributes.position;
            colAttr = new THREE.BufferAttribute(new Float32Array(posAttr.count * 3), 3);
            geo.setAttribute('color', colAttr);
        }
        for(let i=0; i<colAttr.count; i++) colAttr.setXYZ(i, c0.r, c0.g, c0.b);
        colAttr.needsUpdate = true;
        
        mesh.material.color.setHex(0xFFFFFF);
        mesh.material.opacity = 0.5; // Transparent 1 chút
        mesh.material.transparent = true;
        mesh.material.needsUpdate = true;
    });
}


function getHeatmapColor(val) {
    const t = Math.max(0, Math.min(1, val));
    let r = 0, g = 0, b = 0;
    if (t < 0.25) { r = 0; g = 4*t; b = 1; }
    else if (t < 0.5) { r = 0; g = 1; b = 1 - 4*(t - 0.25); }
    else if (t < 0.75) { r = 4*(t - 0.5); g = 1; b = 0; }
    else { r = 1; g = 1 - 4*(t - 0.75); b = 0; }
    return new THREE.Color(Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b)));
}

function updateHeatmapProgressive
(currentVWorld) {
    if (!globalVertexSDFs) {
        globalVertexSDFs = new Map();
        const pTop = new THREE.Vector3(0, 3, 0);
        const pBot = new THREE.Vector3(0, -3, 0);
        const pFront = new THREE.Vector3(0, 0, 3);
        const pRight = new THREE.Vector3(3, 0, 0);
        const pBack = new THREE.Vector3(0, 0, -3);
        const pLeft = new THREE.Vector3(-3, 0, 0);
        const uniqueVerts = [pTop, pBot, pFront, pRight, pBack, pLeft];
        
        for(let vLocal of uniqueVerts) {
            const vWorld = vLocal.clone().applyMatrix4(SC.modelGroup.matrixWorld);
            const inNormal = vLocal.clone().multiplyScalar(-1).normalize().transformDirection(SC.modelGroup.matrixWorld).normalize();
            
            const up = new THREE.Vector3(0, 1, 0);
            if (Math.abs(inNormal.dot(up)) > 0.999) up.set(1, 0, 0);
            const localY = inNormal.clone();
            const localX = new THREE.Vector3().crossVectors(up, localY).normalize();
            const localZ = new THREE.Vector3().crossVectors(localX, localY).normalize();
            const rotMat = new THREE.Matrix4().makeBasis(localX, localY, localZ);
            const q = new THREE.Quaternion().setFromRotationMatrix(rotMat);
            
            const hits = [];
            for(let i=0; i<numPoints; i++){
                const localDir = ptData[i].pInner.clone().normalize();
                const worldDir = localDir.clone().applyQuaternion(q).normalize();
                raycaster.set(vWorld.clone().add(worldDir.clone().multiplyScalar(0.01)), worldDir);
                const intersects = raycaster.intersectObjects(SC.faces, false);
                if(intersects.length > 0) {
                    hits.push(intersects[0].distance + 0.01);
                }
            }
            const valid = hits.filter(d => d < 15);
            const dAvg = valid.length > 0 ? (valid.reduce((a,b)=>a+b, 0) / valid.length) : 0;
            globalVertexSDFs.set(vWorld.toArray().map(v=>v.toFixed(3)).toString(), dAvg);
            if (dAvg < globalMinSDF) globalMinSDF = dAvg;
            if (dAvg > globalMaxSDF) globalMaxSDF = dAvg;
        }
    }
    
    activeVertices.push(currentVWorld.toArray().map(v=>v.toFixed(3)).toString());
    
    SC.faces.forEach(mesh => {
        const geo = mesh.geometry;
        const posAttr = geo.attributes.position;
        let colAttr = geo.attributes.color;
        if (!colAttr) {
            colAttr = new THREE.BufferAttribute(new Float32Array(posAttr.count * 3), 3);
            geo.setAttribute('color', colAttr);
        }
        
        for(let i=0; i<posAttr.count; i++){
            const pLocal = new THREE.Vector3(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i));
            const pWorld = pLocal.clone().applyMatrix4(SC.modelGroup.matrixWorld);
            let bestDist = Infinity;
            let bestSDF = 0;
            let closestKey = null;
            for(let [keyStr, sdf] of globalVertexSDFs.entries()) {
                const keyPt = new THREE.Vector3().fromArray(keyStr.split(',').map(Number));
                const dist = pWorld.distanceTo(keyPt);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestSDF = sdf;
                    closestKey = keyStr;
                }
            }
            
            if (activeVertices.includes(closestKey)) {
                let t = (globalMaxSDF > globalMinSDF) ? (bestSDF - globalMinSDF) / (globalMaxSDF - globalMinSDF) : 0.5;
                const c = getHeatmapColor(t);
                colAttr.setXYZ(i, c.r, c.g, c.b);
            } else {
                const c0 = getHeatmapColor(0);
                colAttr.setXYZ(i, c0.r, c0.g, c0.b);
            }
        }
        colAttr.needsUpdate = true;
        mesh.material.color.setHex(0xFFFFFF);
        mesh.material.opacity = 0.5;
        mesh.material.needsUpdate = true;
    });
}



function ensureRandomRays() {
    const randomCount = ptData.filter(d => !d.isUser).length;
    if (randomCount < 20) {
        const needed = 20 - randomCount;
        for(let i=0; i<needed; i++) {
            const idx = addRayData(Math.random(), Math.random(), false);
            if (SC.isAnimating) {
                ptMeshes[idx].visible = false;
                ptMeshes[idx].material.opacity = 0;
                radialLines[idx].visible = false;
                radialLines[idx].material.opacity = 0;
            }
        }
        updateRayIntersections();
    }
}


function computeFullHeatmap() {
    resetHeatmap();
    const pTop = new THREE.Vector3(0, 3, 0);
    const pBot = new THREE.Vector3(0, -3, 0);
    const pFront = new THREE.Vector3(0, 0, 3);
    const pRight = new THREE.Vector3(3, 0, 0);
    const pBack = new THREE.Vector3(0, 0, -3);
    const pLeft = new THREE.Vector3(-3, 0, 0);
    const uniqueVerts = [pTop, pBot, pFront, pRight, pBack, pLeft];
    for(let vLocal of uniqueVerts) {
        const vWorld = vLocal.clone().applyMatrix4(SC.modelGroup.matrixWorld);
        updateHeatmapProgressive(vWorld);
    }
}

    return {


        pointsGroup, circleLine,
        rays, intersectDots, rayMatObj,
        intersectPts, hitDistances,
        addRayData, updateRayIntersections, resetHeatmap, updateHeatmapProgressive, computeFullHeatmap, ensureRandomRays
    };
}
