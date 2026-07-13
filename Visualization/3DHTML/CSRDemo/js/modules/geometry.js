import * as THREE from 'three';

export function setupGeometry(SC) {
    const { scene } = SC;

    const modelGroup = new THREE.Group();
    scene.add(modelGroup);

    // To make it look nice, add a small tilt
    modelGroup.rotation.set(Math.PI / 8, Math.PI / 6, 0);

    // 6 Vertices of Octahedron
    const pTop = new THREE.Vector3(0, 3, 0);
    const pBot = new THREE.Vector3(0, -3, 0);
    const pFront = new THREE.Vector3(0, 0, 3);
    const pRight = new THREE.Vector3(3, 0, 0);
    const pBack = new THREE.Vector3(0, 0, -3);
    const pLeft = new THREE.Vector3(-3, 0, 0);

    const vertices = [pTop, pBot, pFront, pRight, pBack, pLeft];
    
    // Create node dots
    const nodeMeshes = [];
    vertices.forEach((v, i) => {
        const dot = new THREE.Mesh(
            new THREE.SphereGeometry(0.15, 16, 16),
            new THREE.MeshBasicMaterial({ color: 0xFFFFFF, transparent: true, opacity: 0 })
        );
        dot.position.copy(v);
        modelGroup.add(dot);
        nodeMeshes.push(dot);
    });

    // Create Faces
    const faces = [];
    const faceGeo = new THREE.BufferGeometry();
    // 8 faces of the octahedron
    const faceIndices = [
        [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 2], // Top half
        [1, 3, 2], [1, 4, 3], [1, 5, 4], [1, 2, 5]  // Bottom half
    ];

    const faceMat = new THREE.MeshPhongMaterial({ 
        color: 0x1155AA, 
        side: THREE.DoubleSide, 
        transparent: true, 
        opacity: 0.8 
    });

    faceIndices.forEach(indices => {
        const geo = new THREE.BufferGeometry().setFromPoints([
            vertices[indices[0]], vertices[indices[1]], vertices[indices[2]]
        ]);
        geo.computeVertexNormals();
        const mesh = new THREE.Mesh(geo, faceMat);
        modelGroup.add(mesh);
        faces.push(mesh);
    });

    // Create Edges
    const edges = [];
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x88CCFF, transparent: true, opacity: 0 });
    
    // Unique edges
    const uniqueEdges = [
        [0,2], [0,3], [0,4], [0,5], // Top to equator
        [1,2], [1,3], [1,4], [1,5], // Bottom to equator
        [2,3], [3,4], [4,5], [5,2]  // Equator
    ];

    uniqueEdges.forEach(pair => {
        const geo = new THREE.BufferGeometry().setFromPoints([
            vertices[pair[0]], vertices[pair[1]]
        ]);
        const line = new THREE.Line(geo, edgeMat);
        modelGroup.add(line);
        edges.push({ line, v1: pair[0], v2: pair[1] });
    });

    // Create HTML Labels for Nodes
    const labels = [];
    vertices.forEach((v, i) => {
        const div = document.createElement('div');
        div.className = 'node-label';
        div.textContent = i;
        div.style.position = 'absolute';
        div.style.color = '#00FF00';
        div.style.fontWeight = 'bold';
        div.style.fontSize = '24px';
        div.style.textShadow = '0 0 4px #000';
        div.style.opacity = '0';
        div.style.pointerEvents = 'none';
        document.body.appendChild(div);
        labels.push({ div, pos: v });
    });

    return { modelGroup, vertices, nodeMeshes, faces, edges, labels };
}
