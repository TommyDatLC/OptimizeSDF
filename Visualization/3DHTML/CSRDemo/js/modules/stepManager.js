import * as THREE from 'three';
import TWEEN from '@tweenjs/tween.js';

export function setupStepManager(SC) {
    const { camera, renderer, tweenGroup, modelGroup, vertices, nodeMeshes, faces, edges, labels } = SC;

    const EASE = TWEEN.Easing.Exponential.InOut;
    const ANIM_DURATION = 1500;

    const steps = [
        "Step 0: Initializing...",
        "Step 1: The 3D Model",
        "Step 2: Turn into Graph",
        "Step 3: Adjacency Matrix",
        "Step 4: Compressed Sparse Row (CSR)"
    ];

    // Helper: update labels position in render loop
    const updateLabels = () => {
        const widthHalf = window.innerWidth / 2;
        const heightHalf = window.innerHeight / 2;
        labels.forEach(l => {
            const pos = l.pos.clone();
            pos.applyMatrix4(modelGroup.matrixWorld);
            pos.project(camera);
            
            l.div.style.left = (pos.x * widthHalf + widthHalf) + 'px';
            l.div.style.top = (-(pos.y * heightHalf) + heightHalf) + 'px';
        });
    };
    
    // Add to animation loop
    const oldAnimate = window.requestAnimationFrame;
    let animId;
    const customAnimate = () => {
        animId = oldAnimate(customAnimate);
        updateLabels();
    };
    customAnimate();

    // Data for Matrix and CSR
    const matrix = [
        [0, 0, 1, 1, 1, 1], // Top (0) connects to 2,3,4,5
        [0, 0, 1, 1, 1, 1], // Bot (1) connects to 2,3,4,5
        [1, 1, 0, 1, 0, 1], // Front (2) connects to 0,1,3,5
        [1, 1, 1, 0, 1, 0], // Right (3) connects to 0,1,2,4
        [1, 1, 0, 1, 0, 1], // Back (4) connects to 0,1,3,5
        [1, 1, 1, 0, 1, 0]  // Left (5) connects to 0,1,2,4
    ];

    const generateMatrixDOM = () => {
        const grid = document.getElementById('matrix-grid');
        grid.innerHTML = '';
        
        // Header Row
        grid.appendChild(createCell('', 'matrix-header'));
        for(let j=0; j<6; j++) grid.appendChild(createCell(j, 'matrix-header'));

        for(let i=0; i<6; i++) {
            grid.appendChild(createCell(i, 'matrix-header')); // Row Header
            for(let j=0; j<6; j++) {
                const val = matrix[i][j];
                const cell = createCell(val, val === 1 ? 'matrix-val-1' : 'matrix-val-0');
                cell.id = `cell-${i}-${j}`;
                cell.dataset.val = val;
                cell.dataset.row = i;
                cell.dataset.col = j;
                grid.appendChild(cell);
            }
        }
    };

    const createCell = (text, className) => {
        const div = document.createElement('div');
        div.className = `matrix-cell ${className}`;
        div.textContent = text;
        return div;
    };

    const tweenObj = (obj, target, duration = ANIM_DURATION) => {
        return new Promise(resolve => {
            new TWEEN.Tween(obj)
                .to(target, duration)
                .easing(EASE)
                .onComplete(resolve)
                .start();
        });
    };

    const tweenDOM = (dom, target, duration = ANIM_DURATION) => {
        return new Promise(resolve => {
            // Very simple CSS transition via style
            dom.style.transition = `all ${duration/1000}s cubic-bezier(0.19, 1, 0.22, 1)`;
            Object.assign(dom.style, target);
            setTimeout(resolve, duration);
        });
    };

    const goToStep = async (step) => {
        if (step < 1 || step > 4) return;
        if (SC.isAnimating) return;
        
        SC.isAnimating = true;
        document.getElementById('step-description').textContent = steps[step];

        if (step === 1) {
            // Restore Faces, hide edges, hide labels
            const promises = faces.map(f => tweenObj(f.material, { opacity: 0.8 }));
            promises.push(...edges.map(e => tweenObj(e.line.material, { opacity: 0 })));
            promises.push(...nodeMeshes.map(n => tweenObj(n.material, { opacity: 0 })));
            labels.forEach(l => tweenDOM(l.div, { opacity: 0 }));
            
            tweenDOM(document.getElementById('bottom-panel'), { height: '0vh' });
            tweenObj(camera.position, { x: 0, y: 5, z: 15 });
            
            await Promise.all(promises);
        }
        else if (step === 2) {
            // Hide Faces, Show Edges & Nodes & Labels
            const promises = faces.map(f => tweenObj(f.material, { opacity: 0 }));
            promises.push(...edges.map(e => tweenObj(e.line.material, { opacity: 0.5 })));
            promises.push(...nodeMeshes.map(n => tweenObj(n.material, { opacity: 1 })));
            labels.forEach(l => tweenDOM(l.div, { opacity: 1 }));
            
            tweenDOM(document.getElementById('bottom-panel'), { height: '0vh' });
            
            await Promise.all(promises);
        }
        else if (step === 3) {
            generateMatrixDOM();
            
            // Show split screen
            tweenDOM(document.getElementById('bottom-panel'), { height: '50vh' });
            
            // Hide CSR
            tweenDOM(document.getElementById('csr-container'), { opacity: 0, display: 'none' });
            
            await new Promise(r => setTimeout(r, ANIM_DURATION));
        }
        else if (step === 4) {
            // Morph to CSR
            const csrContainer = document.getElementById('csr-container');
            csrContainer.style.display = 'block';
            await tweenDOM(csrContainer, { opacity: 1 }, 500);

            // Calculate CSR
            const valDiv = document.getElementById('csr-values');
            const colDiv = document.getElementById('csr-columns');
            const ptrDiv = document.getElementById('csr-rowptr');
            
            valDiv.innerHTML = '';
            colDiv.innerHTML = '';
            ptrDiv.innerHTML = '';

            let ptrCount = 0;
            const ptrs = [0];
            
            // Create target cells in CSR arrays (invisible at first)
            const targetCells = [];
            for(let i=0; i<6; i++) {
                for(let j=0; j<6; j++) {
                    if (matrix[i][j] === 1) {
                        const vDiv = createCell('1', 'csr-cell csr-val');
                        vDiv.style.opacity = 0;
                        valDiv.appendChild(vDiv);
                        
                        const cDiv = createCell(j, 'csr-cell csr-col');
                        cDiv.style.opacity = 0;
                        colDiv.appendChild(cDiv);
                        
                        targetCells.push({ i, j, vDiv, cDiv });
                        ptrCount++;
                    }
                }
                ptrs.push(ptrCount);
            }
            
            ptrs.forEach((p, rowIdx) => {
                const pDiv = createCell(p, 'csr-cell csr-ptr');
                pDiv.style.opacity = 0;
                pDiv.style.cursor = 'pointer';
                pDiv.title = `Row ${rowIdx} pointer`;
                ptrDiv.appendChild(pDiv);
                setTimeout(() => pDiv.style.opacity = 1, ANIM_DURATION);

                let isActive = false;
                pDiv.addEventListener('click', () => {
                    // Reset all
                    ptrDiv.childNodes.forEach(node => { node.isActive = false; node.style.opacity = 1; node.style.transform = 'scale(1)'; });
                    valDiv.childNodes.forEach(node => { node.style.opacity = 1; node.style.transform = 'scale(1)'; });
                    colDiv.childNodes.forEach(node => { node.style.opacity = 1; node.style.transform = 'scale(1)'; });
                    for (let i=0; i<6; i++) {
                        for (let j=0; j<6; j++) {
                            const mc = document.getElementById(`cell-${i}-${j}`);
                            if (mc) { mc.style.opacity = 1; mc.style.transform = 'scale(1)'; }
                        }
                    }
                    
                    isActive = !isActive;
                    pDiv.isActive = isActive;
                    
                    if (isActive) {
                        // Dim others
                        ptrDiv.childNodes.forEach(node => { if (node !== pDiv) node.style.opacity = 0.2; });
                        valDiv.childNodes.forEach(node => node.style.opacity = 0.2);
                        colDiv.childNodes.forEach(node => node.style.opacity = 0.2);
                        for (let i=0; i<6; i++) {
                            for (let j=0; j<6; j++) {
                                const mc = document.getElementById(`cell-${i}-${j}`);
                                if (mc) mc.style.opacity = 0.2;
                            }
                        }
                        
                        pDiv.style.transform = 'scale(1.2)';
                        
                        if (rowIdx < ptrs.length - 1) {
                            const start = ptrs[rowIdx];
                            const end = ptrs[rowIdx + 1];
                            for (let k = start; k < end; k++) {
                                if (valDiv.childNodes[k]) {
                                    valDiv.childNodes[k].style.opacity = 1;
                                    valDiv.childNodes[k].style.transform = 'scale(1.2)';
                                }
                                if (colDiv.childNodes[k]) {
                                    colDiv.childNodes[k].style.opacity = 1;
                                    colDiv.childNodes[k].style.transform = 'scale(1.2)';
                                }
                            }
                            // Highlight the corresponding row in the Adjacency Matrix
                            for (let j=0; j<6; j++) {
                                const mc = document.getElementById(`cell-${rowIdx}-${j}`);
                                if (mc) {
                                    mc.style.opacity = 1;
                                    // Highlight non-zero cells more
                                    if (mc.dataset.val === "1") mc.style.transform = 'scale(1.15)';
                                }
                            }
                        } else {
                            // Last element, show all
                            valDiv.childNodes.forEach(node => { node.style.opacity = 1; node.style.transform = 'scale(1)'; });
                            colDiv.childNodes.forEach(node => { node.style.opacity = 1; node.style.transform = 'scale(1)'; });
                            for (let i=0; i<6; i++) {
                                for (let j=0; j<6; j++) {
                                    const mc = document.getElementById(`cell-${i}-${j}`);
                                    if (mc) { mc.style.opacity = 1; mc.style.transform = 'scale(1)'; }
                                }
                            }
                        }
                    }
                });
            });

            // Fly animation
            const flyPromises = [];
            targetCells.forEach(({ i, j, vDiv, cDiv }) => {
                const sourceCell = document.getElementById(`cell-${i}-${j}`);
                if (!sourceCell) return;
                
                const sRect = sourceCell.getBoundingClientRect();
                const vRect = vDiv.getBoundingClientRect();
                
                // Create a clone to fly
                const flyCell = sourceCell.cloneNode(true);
                flyCell.className = 'matrix-cell matrix-val-1 flying-cell';
                flyCell.style.left = sRect.left + 'px';
                flyCell.style.top = sRect.top + 'px';
                document.body.appendChild(flyCell);
                
                // Hide original
                sourceCell.style.opacity = 0.1;
                
                // Fly to val
                flyPromises.push(new Promise(resolve => {
                    setTimeout(() => {
                        flyCell.style.left = vRect.left + 'px';
                        flyCell.style.top = vRect.top + 'px';
                        flyCell.style.backgroundColor = '#00FF00'; // Match CSR color
                        
                        setTimeout(() => {
                            flyCell.remove();
                            vDiv.style.opacity = 1;
                            cDiv.style.opacity = 1;
                            resolve();
                        }, ANIM_DURATION);
                    }, Math.random() * 500); // Stagger start
                }));
            });

            await Promise.all(flyPromises);
        }

        SC.currentStep = step;
        SC.isAnimating = false;
    };

    return { goToStep };
}
