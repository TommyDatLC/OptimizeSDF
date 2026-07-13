import * as THREE from 'three';
import TWEEN from '@tweenjs/tween.js';

export function setupGeometry(SC) {
    const { scene, coneInitialPos } = SC;
    // 2. MÔ HÌNH HÌNH THOI (CÁCH XA TRUNG TÂM)
    // ==========================================
    const modelGroup = new THREE.Group();
    scene.add(modelGroup);

    modelGroup.position.set(7, 2, -4);
    modelGroup.rotation.set(Math.PI / 6, Math.PI / 4, 0);
    modelGroup.scale.set(2, 1, 1);

    const pTop = new THREE.Vector3(0, 3, 0);
    const pBot = new THREE.Vector3(0, -3, 0);
    const pFront = new THREE.Vector3(0, 0, 3);
    const pRight = new THREE.Vector3(3, 0, 0);
    const pBack = new THREE.Vector3(0, 0, -3);
    const pLeft = new THREE.Vector3(-3, 0, 0);

    const faces = [];
    function createTri(p1, p2, p3) {
        const geo = new THREE.BufferGeometry().setFromPoints([p1, p2, p3]);
        geo.computeVertexNormals();
        const colors = new Float32Array([1,1,1, 1,1,1, 1,1,1]);
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        const mat = new THREE.MeshPhongMaterial({ color: 0x1155AA, side: THREE.DoubleSide, transparent: true, opacity: 0.8, vertexColors: true });
        const mesh = new THREE.Mesh(geo, mat);
        const edgeMat = new THREE.LineBasicMaterial({ color: 0x88CCFF, transparent: true, opacity: 0.5 });
        mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat));
        modelGroup.add(mesh);
        faces.push(mesh);
    }

    createTri(pTop, pFront, pRight); createTri(pTop, pRight, pBack);
    createTri(pTop, pBack, pLeft);   createTri(pTop, pLeft, pFront);
    createTri(pBot, pRight, pFront); createTri(pBot, pBack, pRight);
    createTri(pBot, pLeft, pBack);   createTri(pBot, pFront, pLeft);

    const targetVertexLocal = pRight.clone();
    const targetNormalLocal = new THREE.Vector3(1, 0, 0);

    modelGroup.updateMatrixWorld(true);
    const targetVertex = targetVertexLocal.applyMatrix4(modelGroup.matrixWorld);
    const targetNormal = targetNormalLocal.transformDirection(modelGroup.matrixWorld).normalize();
    const invNormal = targetNormal.clone().multiplyScalar(-1);

    const vertexDot = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), new THREE.MeshBasicMaterial({color: 0xFFFFFF}));
    vertexDot.position.copy(targetVertex);
    vertexDot.visible = false;
    scene.add(vertexDot);

    // ==========================================
    // 3. TẠO HÌNH NÓN Ở TRUNG TÂM & ĐƯỜNG TOÁN HỌC (ĐỊNH HƯỚNG THEO TRỤC Y)
    // ==========================================
    const coneHeight = 3.0;
    const coneHalfAngle = 35 * (Math.PI / 180); // Nửa góc nón
    const coneRadius = coneHeight * Math.tan(coneHalfAngle);

    const coneGroup = new THREE.Group();
    coneGroup.position.copy(coneInitialPos);
    scene.add(coneGroup);

    const coneGeo = new THREE.CylinderGeometry(coneRadius, 0.01, coneHeight, 32);
    coneGeo.translate(0, coneHeight / 2, 0);

    const coneMat = new THREE.MeshPhongMaterial({ color: 0x2288FF, transparent: true, opacity: 0.0, side: THREE.DoubleSide, depthWrite: false });
    const coneMesh = new THREE.Mesh(coneGeo, coneMat);
    const coneEdges = new THREE.LineSegments(new THREE.EdgesGeometry(coneGeo), new THREE.LineBasicMaterial({ color: 0x88CCFF, transparent: true, opacity: 0.0 }));
    coneGroup.add(coneMesh);
    coneGroup.add(coneEdges);

    const mathGroup = new THREE.Group();
    const lineMatYellow = new THREE.LineBasicMaterial({ color: 0xFFFF00, linewidth: 3, transparent: true, opacity: 0 });
    const lineMatRed = new THREE.LineBasicMaterial({ color: 0xFF0000, linewidth: 3, transparent: true, opacity: 0 });

    // Trục dọc theo phương Y
    const hLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0, coneHeight, 0)]), lineMatYellow);
    // Bán kính theo phương X
    const rLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, coneHeight, 0), new THREE.Vector3(coneRadius, coneHeight, 0)]), lineMatRed);
    // Cạnh huyền 1 & 2
    const hypLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(coneRadius, coneHeight, 0)]), lineMatYellow);
    const hypLine2 = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(-coneRadius, coneHeight, 0)]), lineMatYellow);

    mathGroup.add(hLine, rLine, hypLine, hypLine2);

    // Vẽ cung tròn hiển thị góc Theta toàn phần (từ -halfAngle đến +halfAngle)
    const arcRadius = 1.2;
    const arcPts = [];
    for (let i = -20; i <= 20; i++) {
        const a = (i / 20) * coneHalfAngle;
        arcPts.push(new THREE.Vector3(arcRadius * Math.sin(a), arcRadius * Math.cos(a), 0));
    }
    const arcLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(arcPts), new THREE.LineBasicMaterial({ color: 0xFFFFFF, linewidth: 2, transparent: true, opacity: 0 }));
    mathGroup.add(arcLine);

    coneGroup.add(mathGroup);

    hLine.visible = false; rLine.visible = false; hypLine.visible = false; hypLine2.visible = false; arcLine.visible = false;

    const hMidPos = new THREE.Vector3(0, coneHeight / 2, 0);
    // Đặt nhãn theta thấp xuống một chút để không trùng với số 1 (Y=1.5)
    const thetaPos = new THREE.Vector3(0, arcRadius - 0.3, 0);

    const thetaLabel = document.getElementById('theta-label');
    katex.render("\\theta", thetaLabel, { throwOnError: false });

    // ==========================================

    // 5. CROSS PRODUCT VÀ MA TRẬN TRỰC GIAO
    // ==========================================
    const worldUp = new THREE.Vector3(0, 1, 0);
    if (Math.abs(invNormal.dot(worldUp)) > 0.999) worldUp.set(1, 0, 0);

    const localY = invNormal.clone();
    const localX = new THREE.Vector3().crossVectors(worldUp, localY).normalize();
    const localZ = new THREE.Vector3().crossVectors(localX, localY).normalize();

    const rotMatrix = new THREE.Matrix4().makeBasis(localX, localY, localZ);
    const finalQuat = new THREE.Quaternion().setFromRotationMatrix(rotMatrix);

    function createArrow(origin, dir, length, colorHex) {
        const arrow = new THREE.ArrowHelper(dir.clone().normalize(), origin, length, colorHex, length*0.2, length*0.15);
        arrow.visible = false;
        scene.add(arrow);
        return arrow;
    }

    const lenVec = 2.0;
    const arrNorm = createArrow(targetVertex, targetNormal, lenVec, 0x888888);
    const arrInvNorm = createArrow(targetVertex, localY, lenVec, 0x00FFFF);
    const arrUp = createArrow(targetVertex, worldUp, lenVec, 0xFFFFFF);
    const arrRight = createArrow(targetVertex, localX, lenVec, 0xFF0000);
    const arrTrueUp = createArrow(targetVertex, localZ, lenVec, 0x00FF00);

    // ==========================================

    return {
        modelGroup, coneGroup, mathGroup,
        hLine, rLine, hypLine, hypLine2, arcLine,
        coneMesh, coneEdges,
        targetVertex, targetNormal, invNormal, vertexDot,
        arrNorm, arrInvNorm, arrUp, arrRight, arrTrueUp,
        finalQuat, coneHeight, coneRadius,
        faces, createArrow, hMidPos, thetaPos
    };
}
