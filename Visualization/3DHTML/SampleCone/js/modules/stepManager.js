import * as THREE from 'three';

export function setupStepManager(SC) {
    if (!window.tweenGroup) window.tweenGroup = new window.TWEEN.Group();

    const { scene, camera, renderer, controls, coneGroup, mathGroup, pointsGroup, modelGroup, hLine, rLine, hypLine, hypLine2, arcLine, coneMesh, coneEdges, targetVertex, targetNormal, invNormal, vertexDot, arrNorm, arrInvNorm, arrUp, arrRight, arrTrueUp, coneInitialPos, startCamPos, finalQuat, coneHeight, coneRadius, tweenArrow, updateRayIntersections, resetHeatmap, updateHeatmapProgressive, computeFullHeatmap, ensureRandomRays, faces, rays, intersectDots, circleLine, hitDistances, intersectPts, hMidPos, thetaPos, addRayData } = SC;
    const ptData = window.ptData;
    const ptMeshes = window.ptMeshes;
    const radialLines = window.radialLines;
    const lenVec = 3;
    
    const mathFormulas = [
        "", // Step 0
        "\\text{Step 1:} \\quad y = R [\\cos\\theta_{max} + (1 - \\cos\\theta_{max}) u_1]", // Step 1
        "\\text{Step 2:} \\quad (x, z) = \\sqrt{R^2 - y^2} \\cdot (\\cos 2\\pi u_2, \\sin 2\\pi u_2)", // Step 2
        "\\text{Step 3:} \\quad \\text{Auto- Generate N rays}", // Step 3
        "\\text{Step 4:} \\quad \\vec{N}_{vertex}", // Step 4
        "\\text{Step 5:} \\quad {\\color{#00FFFF}\\vec{Y}} = {\\color{#00FFFF}\\vec{N}_{inv}} = -\\vec{N}_{vertex}", // Step 5
        "\\text{Step 6:} \\quad \\vec{Up} = (0, 1, 0)", // Step 6
        "\\text{Step 7:} \\quad {\\color{#FF0000}\\vec{X}} = \\vec{Up} \\times {\\color{#00FFFF}\\vec{Y}}", // Step 7
        "\\text{Step 8:} \\quad {\\color{#FF0000}\\vec{X}} = \\vec{Up} \\times {\\color{#00FFFF}\\vec{Y}}", // Step 8
        "\\text{Step 9:} \\quad {\\color{#00FF00}\\vec{Z}} = {\\color{#FF0000}\\vec{X}} \\times {\\color{#00FFFF}\\vec{Y}}", // Step 9
        "\\text{Step 10:} \\quad M_{rot} = [{\\color{#FF0000}\\vec{X}}, {\\color{#00FFFF}\\vec{Y}}, {\\color{#00FF00}\\vec{Z}}]", // Step 10
        "\\text{Step 11:} \\quad M_{rot} \\times \\text{Random Ray}_{rand}", // Step 11
        "\\text{Step 12:} \\quad Ray_{i\\_rand} \\to \\infty", // Step 12
        `\\text{Step 13:} \\quad d_{avg} = \\frac{1}{${ptData.length}} \\sum d_{i\\_rand}`, // Step 13
        "\\text{Step 14:} \\quad SDF_{target} = d_{avg}", // Step 14
        "\\text{Step 15:} \\quad \\text{Loop over Vertices \\& Compute SDF}" // Step 15
    ];

    const mathOverlay = document.getElementById('math-formula');
    const hLabel = document.getElementById('h-label');
    const thetaLabel = document.getElementById('theta-label');

    function updateMathFormulaImmediate(step) {
        if (step === 0) {
            mathOverlay.style.opacity = 0;
            mathOverlay.style.display = 'none';
        } else {
            mathOverlay.style.display = 'block';
            katex.render(mathFormulas[step], mathOverlay, { throwOnError: false });
            mathOverlay.style.opacity = 1;
        }
    }

    async function changeMathFormula(step) {
        if (mathOverlay.style.opacity > 0 && mathOverlay.style.display !== 'none') {
            mathOverlay.style.opacity = 0;
            await new Promise(r => setTimeout(r, 300));
        }
        if (step === 0) {
            mathOverlay.style.display = 'none';
        } else {
            mathOverlay.style.display = 'block';
            katex.render(mathFormulas[step], mathOverlay, { throwOnError: false });
            mathOverlay.style.opacity = 1;
        }
    }

    function updateLabels() {
        if (window.currentStep === 1) {
            hLabel.style.display = 'block';
            thetaLabel.style.display = 'block';

            const v1 = hMidPos.clone().applyMatrix4(coneGroup.matrixWorld).project(camera);
            hLabel.style.left = `${(v1.x * 0.5 + 0.5) * window.innerWidth + 15}px`;
            hLabel.style.top = `${(v1.y * -0.5 + 0.5) * window.innerHeight}px`;

            const vTheta = thetaPos.clone().applyMatrix4(coneGroup.matrixWorld).project(camera);
            thetaLabel.style.left = `${(vTheta.x * 0.5 + 0.5) * window.innerWidth}px`;
            thetaLabel.style.top = `${(vTheta.y * -0.5 + 0.5) * window.innerHeight}px`;
        } else {
            hLabel.style.display = 'none';
            thetaLabel.style.display = 'none';
        }
    }

    const MAX_STEPS = 15;
    window.currentStep = 0;
    let isAnimating = false; window.isAnimating = false;

        function tweenObj(obj, props, duration, easing = window.TWEEN.Easing.Quadratic.Out, onUpdate = null) {
        
        return new Promise(resolve => {
            const tween = new window.TWEEN.Tween(obj, window.tweenGroup).to(props, duration).easing(easing).onComplete(() => {
                
                resolve();
            });
            if (onUpdate) tween.onUpdate(onUpdate);
            tween.start();
        });
    }

    function updatePanelVisibility(step) {
        const p1 = document.getElementById('panel-step1');
        const p2 = document.getElementById('panel-step2');
        if (p1) p1.style.display = (step === 1) ? 'block' : 'none';
        if (p2) p2.style.display = (step === 2) ? 'block' : 'none';
    }

    function snapToStep(step, snapCamera = true) {
        updatePanelVisibility(step);
        
        if (step >= 3 && ptData.length < 20) {
            const needed = 20 - ptData.length;
            for(let i=0; i<needed; i++) addRayData(Math.random(), Math.random(), false);
        }
        
        ptData.forEach(d => {
            if (d.circleVis) d.circleVis.visible = (step === 1);
        });

        updateMathFormulaImmediate(step);

        if (step >= 11 && step < 14) {
            coneGroup.position.copy(targetVertex);
            coneGroup.quaternion.copy(finalQuat);
        } else {
            coneGroup.position.copy(coneInitialPos);
            coneGroup.quaternion.identity();
        }

        coneMesh.material.opacity = (step >= 1 && step < 12) ? 0.3 : 0;
        coneEdges.material.opacity = (step >= 1 && step < 12) ? 0.7 : 0;

        [hLine, rLine, hypLine, hypLine2, arcLine].forEach(l => {
            l.visible = (step === 1);
            l.material.opacity = (step === 1) ? 1 : 0;
        });

        ptMeshes.forEach((pt, i) => {
            const isUser = ptData[i] && ptData[i].isUser;
            let isVisible = false;
            if (isUser) {
                isVisible = (step === 1 || step === 2);
            } else {
                isVisible = (step >= 3 && step < 14);
            }
            pt.visible = isVisible;
            pt.material.opacity = isVisible ? 1 : 0;
            pt.scale.set(1, 1, 1);
            if (isVisible && ptData[i]) {
                pt.position.copy(ptData[i].pInner);
            }
        });

        radialLines.forEach((rl, i) => {
            const isUser = ptData[i] && ptData[i].isUser;
            let isVisible = false;
            if (isUser) {
                isVisible = (step === 1 || step === 2);
            } else {
                isVisible = (step >= 3 && step < 13);
            }
            rl.visible = isVisible;
            rl.material.opacity = isVisible ? 0.8 : 0;
        });

        vertexDot.visible = (step >= 4);
        vertexDot.scale.set(1, 1, 1);

        arrNorm.visible = (step === 4);
        if (step === 4) arrNorm.setLength(lenVec, lenVec*0.2, lenVec*0.15);

        arrInvNorm.visible = (step >= 5);
        if (step >= 5) arrInvNorm.setLength(lenVec, lenVec*0.2, lenVec*0.15);

        arrUp.visible = (step === 6 || step === 7);
        if (step === 6 || step === 7) arrUp.setLength(lenVec, lenVec*0.2, lenVec*0.15);

        arrRight.visible = (step >= 7);
        if (step >= 7) arrRight.setLength(lenVec, lenVec*0.2, lenVec*0.15);

        arrTrueUp.visible = (step >= 9);
        if (step >= 9) arrTrueUp.setLength(lenVec, lenVec*0.2, lenVec*0.15);

        rays.forEach((r, i) => {
            r.visible = (step >= 12 && step < 14);
            if (step >= 12) {
                r.material.opacity = 0.8;
                const startPt = targetVertex;
                let endPt;
                if (step === 12) {
                    const dir = ptData[i].pInner.clone().normalize().applyQuaternion(finalQuat);
                    endPt = startPt.clone().add(dir.multiplyScalar(50));
                } else {
                    endPt = intersectPts[i];
                }
                r.geometry.setFromPoints([startPt, endPt]);
            }
        });

        intersectDots.forEach((dot, i) => {
            if(step >= 13 && hitDistances[i] < 15) {
                dot.visible = true;
                dot.position.copy(intersectPts[i]);
                dot.scale.set(1,1,1);
            } else {
                dot.visible = false;
            }
        });

        // Camera Snap Logic
        if (snapCamera) {
            if (step >= 14) {
                const boxCenter = modelGroup.position.clone();
                controls.target.copy(boxCenter);
                camera.position.copy(boxCenter.clone().add(new THREE.Vector3(-3, 4, 12)));
            } else if (step >= 11) {
                controls.target.copy(targetVertex);
                camera.position.set(targetVertex.x - 5, targetVertex.y + 2, targetVertex.z + 8);
            } else if (step >= 10) {
                controls.target.copy(targetVertex);
                camera.position.set(targetVertex.x + 6, targetVertex.y + 4, targetVertex.z + 4);
            } else if (step >= 4) {
                controls.target.copy(targetVertex);
                camera.position.set(targetVertex.x + 3, targetVertex.y + 3, targetVertex.z + 8);
            } else if (step >= 2 && step <= 3) {
                controls.target.set(0, coneHeight, 0);
                camera.position.set(0.1, 12, 0.1);
            } else {
                controls.target.copy(coneInitialPos);
                camera.position.copy(startCamPos);
            }
        }
        
        if (step === 15) {
            if (window.SC && window.SC.computeFullHeatmap) window.SC.computeFullHeatmap(); else computeFullHeatmap();
        } else if (step === 14) {
            if (window.SC && window.SC.resetHeatmap) window.SC.resetHeatmap();
            if (window.SC && window.SC.updateHeatmapProgressive) window.SC.updateHeatmapProgressive(targetVertex);
            rays.forEach(r => r.visible = false);
            intersectDots.forEach(d => d.visible = false);
        } else {
            faces.forEach(mesh => {
                const colAttr = mesh.geometry.attributes.color;
                if (colAttr) {
                    for(let i=0; i<colAttr.count; i++) colAttr.setXYZ(i, 1, 1, 1);
                    colAttr.needsUpdate = true;
                }
                mesh.material.color.setHex(0x1155AA);
                mesh.material.opacity = 0.8;
                mesh.material.transparent = true;
                mesh.material.needsUpdate = true;
            });
        }
    }

    async function animateTransition(step) {
        updatePanelVisibility(step);
        changeMathFormula(step);

        if (step === 1) {
            [hLine, rLine, hypLine, hypLine2].forEach(l => l.visible = true);
            await Promise.all([
                tweenObj(coneMesh.material, { opacity: 0.3 }, 800),
                tweenObj(coneEdges.material, { opacity: 0.7 }, 800),
                tweenObj(hLine.material, { opacity: 1 }, 800)
            ]);
            // If rays were added, fade them in
            ptMeshes.forEach((pt, i) => {
                if (ptData[i] && ptData[i].isUser) {
                    pt.visible = true;
                    tweenObj(pt.material, { opacity: 1 }, 600);
                } else {
                    pt.visible = false;
                }
            });
            radialLines.forEach((rl, i) => {
                if (ptData[i] && ptData[i].isUser) {
                    rl.visible = true;
                    tweenObj(rl.material, { opacity: 0.8 }, 600);
                } else {
                    rl.visible = false;
                }
            });
        }
        else if (step === 2) {
            // Hide math lines from step 1
            [hLine, rLine, hypLine, hypLine2, arcLine].forEach(l => {
                tweenObj(l.material, { opacity: 0 }, 500).then(()=> l.visible = false);
            });
            // Ensure user rays are explicitly visible and fully opaque
            ptMeshes.forEach((pt, i) => {
                if (ptData[i] && ptData[i].isUser) {
                    pt.visible = true;
                    pt.material.opacity = 1;
                }
            });
            radialLines.forEach((rl, i) => {
                if (ptData[i] && ptData[i].isUser) {
                    rl.visible = true;
                    rl.material.opacity = 0.8;
                }
            });
            await new Promise(r => setTimeout(r, 500));
        }
        else if (step === 3) {
            ensureRandomRays();
            
            // Force THREE.js to compile shaders BEFORE starting tweens
            ptMeshes.forEach(pt => pt.visible = true);
            radialLines.forEach(rl => rl.visible = true);
            renderer.compile(scene, camera);
            
            // Hide them back so they can animate properly
            ptMeshes.forEach((pt, i) => {
                if (ptData[i] && !ptData[i].isUser && pt.material.opacity === 0) pt.visible = false;
            });
            radialLines.forEach((rl, i) => {
                if (ptData[i] && !ptData[i].isUser && rl.material.opacity === 0) rl.visible = false;
            });
            
            window.isCameraPanning = true;
            const panTarget = tweenObj(controls.target, {x: 0, y: coneHeight, z: 0}, 1500, TWEEN.Easing.Cubic.InOut);
            const panCam = tweenObj(camera.position, {x: 0.1, y: 12, z: 0.1}, 1500, TWEEN.Easing.Cubic.InOut);
            
            // Hide user points
            ptMeshes.forEach((pt, i) => {
                if (ptData[i] && ptData[i].isUser) {
                    tweenObj(pt.material, { opacity: 0 }, 500).then(() => pt.visible = false);
                }
            });
            radialLines.forEach((rl, i) => {
                if (ptData[i] && ptData[i].isUser) {
                    tweenObj(rl.material, { opacity: 0 }, 500).then(() => rl.visible = false);
                }
            });
            
            // Show random points
            ptMeshes.forEach((pt, i) => {
                if (ptData[i] && !ptData[i].isUser) {
                    pt.visible = true;
                    pt.scale.set(0.01, 0.01, 0.01);
                    tweenObj(pt.scale, {x:1, y:1, z:1}, 600, TWEEN.Easing.Back.Out);
                    tweenObj(pt.material, {opacity:1}, 600);
                }
            });
            radialLines.forEach((rl, i) => {
                if (ptData[i] && !ptData[i].isUser) {
                    rl.visible = true;
                    rl.material.opacity = 0;
                    tweenObj(rl.material, { opacity: 0.8 }, 600);
                }
            });
            await Promise.all([panTarget, panCam]).then(() => { window.isCameraPanning = false; });
        }
        else if (step === 4) {
            window.isCameraPanning = true;
            const panTarget = tweenObj(controls.target, {x: targetVertex.x, y: targetVertex.y, z: targetVertex.z}, 1500, TWEEN.Easing.Cubic.InOut);
            const panCam = tweenObj(camera.position, {x: targetVertex.x + 3, y: targetVertex.y + 3, z: targetVertex.z + 8}, 1500, TWEEN.Easing.Cubic.InOut);

            vertexDot.visible = true;
            vertexDot.scale.set(0.01, 0.01, 0.01);
            tweenObj(vertexDot.scale, {x:1, y:1, z:1}, 1000, TWEEN.Easing.Back.Out);

            const promisesArray = [
                Promise.all([panTarget, panCam]).then(() => { window.isCameraPanning = false; })
            ];
            await Promise.all(promisesArray);

            arrNorm.visible = true;
            arrNorm.setLength(0.01, 0, 0);
            await tweenObj({l:0}, {l:lenVec}, 600, TWEEN.Easing.Quadratic.Out, obj => arrNorm.setLength(obj.l, obj.l*0.2, obj.l*0.15));
        }
        else if (step === 5) {
            arrInvNorm.visible = true;
            arrInvNorm.setLength(0.01, 0, 0);

            await Promise.all([
                tweenObj({l:lenVec}, {l:0.01}, 600, TWEEN.Easing.Quadratic.Out, obj => arrNorm.setLength(obj.l, obj.l*0.2, obj.l*0.15)),
                tweenObj({l:0}, {l:lenVec}, 600, TWEEN.Easing.Quadratic.Out, obj => arrInvNorm.setLength(obj.l, obj.l*0.2, obj.l*0.15))
            ]).then(() => arrNorm.visible = false);
        }
        else if (step === 6) {
            arrUp.visible = true; arrUp.setLength(0.01, 0, 0);
            await tweenObj({l:0}, {l:lenVec}, 600, TWEEN.Easing.Quadratic.Out, obj => arrUp.setLength(obj.l, obj.l*0.2, obj.l*0.15));
        }
        else if (step === 7) {
            arrRight.visible = true; arrRight.setLength(0.01, 0, 0);
            await tweenObj({l:0}, {l:lenVec}, 600, TWEEN.Easing.Quadratic.Out, obj => arrRight.setLength(obj.l, obj.l*0.2, obj.l*0.15));
        }
        else if (step === 8) {
            await tweenObj({l:lenVec}, {l:0.01}, 400, TWEEN.Easing.Quadratic.In, obj => arrUp.setLength(obj.l, obj.l*0.2, obj.l*0.15))
                .then(() => arrUp.visible = false);
        }
        else if (step === 9) {
            arrTrueUp.visible = true; arrTrueUp.setLength(0.01, 0, 0);
            await tweenObj({l:0}, {l:lenVec}, 600, TWEEN.Easing.Quadratic.Out, obj => arrTrueUp.setLength(obj.l, obj.l*0.2, obj.l*0.15));
        }
        else if (step === 10) {
            await new Promise(r => setTimeout(r, 600));
        }
        else if (step === 11) {
            window.isCameraPanning = true;
            const panTarget = tweenObj(controls.target, {x: targetVertex.x, y: targetVertex.y, z: targetVertex.z}, 1200, TWEEN.Easing.Cubic.InOut);
            const panCam = tweenObj(camera.position, {x: targetVertex.x + 6, y: targetVertex.y + 4, z: targetVertex.z + 4}, 1200, TWEEN.Easing.Cubic.InOut);

            const qStart = coneGroup.quaternion.clone();
            const qTween = tweenObj({ t: 0 }, { t: 1 }, 1200, TWEEN.Easing.Cubic.InOut, obj => coneGroup.quaternion.slerpQuaternions(qStart, finalQuat, obj.t));
            const pTween = tweenObj(coneGroup.position, {x: targetVertex.x, y: targetVertex.y, z: targetVertex.z}, 1200, TWEEN.Easing.Cubic.InOut);

            await Promise.all([pTween, qTween, panTarget, panCam]).then(() => { window.isCameraPanning = false; });
        }
        else if (step === 12) {
            window.isCameraPanning = true;
            const panTarget = tweenObj(controls.target, {x: targetVertex.x, y: targetVertex.y, z: targetVertex.z}, 1200, TWEEN.Easing.Cubic.InOut);
            const panCam = tweenObj(camera.position, {x: targetVertex.x - 5, y: targetVertex.y + 2, z: targetVertex.z + 8}, 1200, TWEEN.Easing.Cubic.InOut);

            tweenObj(coneMesh.material, { opacity: 0 }, 800);
            tweenObj(coneEdges.material, { opacity: 0 }, 800);
            radialLines.forEach(rl => tweenObj(rl.material, { opacity: 0 }, 800).then(() => rl.visible = false));

            const promises = rays.map((r, i) => {
                r.visible = true;
                r.material.opacity = 0.8;
                const dir = ptData[i].pInner.clone().normalize().applyQuaternion(finalQuat);
                const startPt = targetVertex;
                const endPt = startPt.clone().add(dir.multiplyScalar(50));
                return tweenObj({ t: 0 }, { t: 1 }, 1200, TWEEN.Easing.Cubic.Out, obj => {
                    const currentEnd = startPt.clone().lerp(endPt, obj.t);
                    r.geometry.setFromPoints([startPt, currentEnd]);
                });
            });

            promises.push(panTarget, panCam);
            await Promise.all(promises).then(() => { window.isCameraPanning = false; });
        }
        else if (step === 13) {
            ptMeshes.forEach(pt => tweenObj(pt.material, { opacity: 0 }, 500).then(() => pt.visible = false));

            const promises = rays.map((r, i) => {
                const startPt = targetVertex;
                const dir = ptData[i].pInner.clone().normalize().applyQuaternion(finalQuat);
                const infPt = startPt.clone().add(dir.multiplyScalar(50));
                const hitPt = intersectPts[i];

                if (hitDistances[i] < 15) {
                    intersectDots[i].position.copy(hitPt);
                    intersectDots[i].scale.set(0.01, 0.01, 0.01);
                    intersectDots[i].visible = true;
                    tweenObj(intersectDots[i].scale, {x:1, y:1, z:1}, 500, TWEEN.Easing.Back.Out);
                }

                return tweenObj({ t: 0 }, { t: 1 }, 800, TWEEN.Easing.Cubic.Out, obj => {
                    const currentEnd = infPt.clone().lerp(hitPt, obj.t);
                    r.geometry.setFromPoints([startPt, currentEnd]);
                });
            });
            await Promise.all(promises);
        }
        else if (step === 14) {
            document.body.style.pointerEvents = 'none'; // Lock interaction
            
            const promises = rays.map((r, i) => {
                if (!intersectPts[i]) return Promise.resolve();
                return tweenObj({ t: 0 }, { t: 1 }, 1000, TWEEN.Easing.Cubic.InOut, obj => {
                    const currentEnd = intersectPts[i].clone().lerp(targetVertex, obj.t);
                    r.geometry.setFromPoints([targetVertex, currentEnd]);
                    if (intersectDots[i]) intersectDots[i].position.copy(currentEnd);
                }).then(() => {
                    r.visible = false;
                    if (intersectDots[i]) intersectDots[i].visible = false;
                });
            });
            await Promise.all(promises);
            
            if (window.SC && window.SC.resetHeatmap) window.SC.resetHeatmap();
            if (window.SC && window.SC.updateHeatmapProgressive) window.SC.updateHeatmapProgressive(targetVertex);
            
            await new Promise(r => setTimeout(r, 800));
            document.body.style.pointerEvents = 'auto'; // Unlock interaction
        }
        else if (step === 15) {
            document.body.style.pointerEvents = 'none'; // Lock interaction
            
            // Pan camera to view the entire box before jumping
            window.isCameraPanning = true;
            const boxCenter = modelGroup.position.clone();
            const camPos = boxCenter.clone().add(new THREE.Vector3(-3, 4, 12));
            const panTarget = tweenObj(controls.target, {x: boxCenter.x, y: boxCenter.y, z: boxCenter.z}, 1000, TWEEN.Easing.Cubic.InOut);
            const panCam = tweenObj(camera.position, {x: camPos.x, y: camPos.y, z: camPos.z}, 1000, TWEEN.Easing.Cubic.InOut);
            await Promise.all([panTarget, panCam]).then(() => { window.isCameraPanning = false; });
            
            const pTop = new THREE.Vector3(0, 3, 0);
            const pBot = new THREE.Vector3(0, -3, 0);
            const pFront = new THREE.Vector3(0, 0, 3);
            const pRight = new THREE.Vector3(3, 0, 0);
            const pBack = new THREE.Vector3(0, 0, -3);
            const pLeft = new THREE.Vector3(-3, 0, 0);
            const uniqueVerts = [pTop, pBot, pFront, pRight, pBack, pLeft];
            
            if (window.SC && window.SC.resetHeatmap) window.SC.resetHeatmap();

            for(let vLocal of uniqueVerts) {
                if (!window.isAnimating) break;
                const vWorld = vLocal.clone().applyMatrix4(modelGroup.matrixWorld);
                
                // --- Detailed logs ---
                console.log(`\n--- Vòng lặp Step 15 ---`);
                console.log(`[Step 15 Loop] Đỉnh Cone mục tiêu (vWorld): ${vWorld.x.toFixed(3)}, ${vWorld.y.toFixed(3)}, ${vWorld.z.toFixed(3)}`);
                
                if (window.SC && window.SC.updateHeatmapProgressive) {
                    window.SC.updateHeatmapProgressive(vWorld);
                }

                const inNormal = vLocal.clone().multiplyScalar(-1).normalize().transformDirection(modelGroup.matrixWorld).normalize();
                
                const up = new THREE.Vector3(0, 1, 0);
                if (Math.abs(inNormal.dot(up)) > 0.999) up.set(1, 0, 0);
                const localY = inNormal.clone();
                const localX = new THREE.Vector3().crossVectors(up, localY).normalize();
                const localZ = new THREE.Vector3().crossVectors(localX, localY).normalize();
                const rotMat = new THREE.Matrix4().makeBasis(localX, localY, localZ);
                const q = new THREE.Quaternion().setFromRotationMatrix(rotMat);
                
                coneGroup.position.copy(vWorld);
                coneGroup.quaternion.copy(q);
                coneGroup.updateMatrixWorld(true);
                
                console.log(`[Step 15 Loop] Đỉnh Cone thực tế (coneGroup.position): ${coneGroup.position.x.toFixed(3)}, ${coneGroup.position.y.toFixed(3)}, ${coneGroup.position.z.toFixed(3)}`);
                console.log(`[Step 15 Loop] Khớp vị trí đỉnh? ${coneGroup.position.distanceTo(vWorld) < 0.001 ? "CÓ" : "KHÔNG"}`);
                
                coneMesh.material.opacity = 0.3;
                coneEdges.material.opacity = 0.7;
                
                const raycaster = new THREE.Raycaster();
                rays.forEach((r, i) => {
                    if (!window.ptData[i]) return;
                    const localDir = window.ptData[i].pInner.clone().normalize();
                    const worldDir = localDir.clone().applyQuaternion(q).normalize();
                    raycaster.set(vWorld.clone().add(worldDir.clone().multiplyScalar(0.01)), worldDir);
                    const intersects = raycaster.intersectObjects(faces, false);
                    let hitPt;
                    if(intersects.length > 0) {
                        hitPt = intersects[0].point;
                    } else {
                        hitPt = vWorld.clone().add(worldDir.clone().multiplyScalar(15));
                    }
                    r.geometry.setFromPoints([vWorld, hitPt]);
                    r.visible = true;
                    intersectDots[i].position.copy(hitPt);
                    intersectDots[i].visible = true;
                });
                
                await new Promise(r => setTimeout(r, 600));
            }
            if (window.isAnimating && window.SC && window.SC.computeFullHeatmap) window.SC.computeFullHeatmap();
            
            coneMesh.material.opacity = 0;
            coneEdges.material.opacity = 0;
            coneGroup.position.copy(coneInitialPos);
            rays.forEach(r => r.visible = false);
            intersectDots.forEach(d => d.visible = false);
            
            if (window.SC && window.SC.computeFullHeatmap) window.SC.computeFullHeatmap(); else computeFullHeatmap();
            
            document.body.style.pointerEvents = 'auto'; // Unlock interaction
        }
    }

    async function goToStep(step) {
        step = Math.max(0, Math.min(step, MAX_STEPS));
        if (step === window.currentStep && !isAnimating) return;

        console.log(`[Transition] Moving from Step ${window.currentStep} to Step ${step}`);
        console.log(`[Camera Start] Pos: ${camera.position.x.toFixed(2)}, ${camera.position.y.toFixed(2)}, ${camera.position.z.toFixed(2)} | Target: ${controls.target.x.toFixed(2)}, ${controls.target.y.toFixed(2)}, ${controls.target.z.toFixed(2)}`);

        TWEEN.removeAll();
        window.isCameraPanning = false;

        if (step < window.currentStep) {
            window.currentStep = step;
            snapToStep(window.currentStep, true);
            isAnimating = false; window.isAnimating = false;
        } else {
            if (isAnimating) {
                snapToStep(window.currentStep, true);
                isAnimating = false; window.isAnimating = false;
                return;
            }
            isAnimating = true; window.isAnimating = true;
            // Không snap camera khi tiến, trượt mượt mà từ vị trí người dùng đang ngắm
            snapToStep(window.currentStep, false);
            window.currentStep = step;
            await animateTransition(window.currentStep);
            isAnimating = false; window.isAnimating = false;
            console.log(`[Camera End] Pos: ${camera.position.x.toFixed(2)}, ${camera.position.y.toFixed(2)}, ${camera.position.z.toFixed(2)} | Target: ${controls.target.x.toFixed(2)}, ${controls.target.y.toFixed(2)}, ${controls.target.z.toFixed(2)}`);
        }
    }

    return { goToStep, snapToStep, animateTransition, changeMathFormula, mathFormulas, MAX_STEPS, tweenObj, updateLabels };
}
