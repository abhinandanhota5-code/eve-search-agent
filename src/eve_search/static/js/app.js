/**
 * EVE Search Agent — WALL-E Mission Control
 * Real-time interactive mission dashboard connecting to the Python simulator.
 */

document.addEventListener('DOMContentLoaded', function () {
    // --- Application State ---
    let currentState = null;
    let isPlaying = false;
    let pollIntervalId = null;
    let heatmapVisible = true;
    let debugMode = false;
    let activeTooltip = null;

    // --- DOM Elements ---
    const gridContainer = document.getElementById('grid-container');
    const logContainer = document.getElementById('log-container');
    const missionConfig = document.getElementById('mission-config');
    const actionFeedback = document.getElementById('action-feedback');
    const viewInfo = document.getElementById('view-info');
    const legendEve = document.getElementById('legend-eve');

    // Controls
    const playPauseBtn = document.getElementById('play-pause');
    const stepBtn = document.getElementById('step');
    const restartBtn = document.getElementById('restart');
    const loadDefaultBtn = document.getElementById('load-default');
    const newSeedBtn = document.getElementById('new-seed');
    const clearLogBtn = document.getElementById('clear-log');
    const heatmapToggle = document.getElementById('heatmap-toggle');
    const debugModeInput = document.getElementById('debug-mode');

    // Stats
    const agentMode = document.getElementById('agent-mode');
    const targetTile = document.getElementById('target-tile');
    const distSafety = document.getElementById('dist-safety');
    const rechargeCount = document.getElementById('recharge-count');
    const batteryLevel = document.getElementById('battery-level');
    const batteryFill = document.getElementById('battery-fill');
    const timeLeft = document.getElementById('time-left');

    // Belief Summary
    const beliefLocation = document.getElementById('belief-location');
    const beliefConfidence = document.getElementById('belief-confidence');
    const beliefEntropy = document.getElementById('belief-entropy');
    const beliefCellsKnown = document.getElementById('belief-cells-known');
    const statusTags = document.getElementById('status-tags');

    // Orientation glyphs for WALL-E
    const ORIENTATION_GLYPHS = {
        'N': '▲',
        'E': '▶',
        'S': '▼',
        'W': '◀'
    };

    // --- Initial Startup ---
    initDashboard();

    async function initDashboard() {
        showFeedback('Connecting to EVE Search Agent simulator...', 'info');
        try {
            // 1. Fetch default configuration text
            const configResp = await fetch('/config/default');
            if (configResp.ok) {
                const configData = await configResp.json();
                if (configData.yaml) {
                    missionConfig.value = configData.yaml;
                }
            }

            // 2. Initialize simulator session with the default config
            await resetMission();
        } catch (err) {
            showFeedback(`Initialization failed: ${err.message}`, 'error');
            console.error('Initialization error:', err);
        }
    }

    // --- Simulator API Operations ---

    async function resetMission() {
        stopPolling();
        isPlaying = false;
        playPauseBtn.textContent = 'Play';
        playPauseBtn.classList.add('btn-primary');

        const configText = missionConfig.value.trim();
        showFeedback('Initializing fresh simulation session...', 'info');

        try {
            const resp = await fetch('/init', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config: configText || 'configs/sample_mission.yaml',
                    include_eval: debugMode
                })
            });

            const data = await resp.json();
            if (!resp.ok || data.status === 'error') {
                throw new Error(data.message || 'Failed to initialize');
            }

            currentState = data.state;
            renderState(currentState);
            addLogEntry(
                '00',
                'INIT',
                currentState.pos,
                currentState.battery,
                currentState.time_left,
                'System ready. Observation barrier active.'
            );
            showFeedback('Mission session initialized and ready.', 'success');
        } catch (err) {
            showFeedback(`Error: ${err.message}`, 'error');
            console.error('Reset error:', err);
        }
    }

    async function executeStep(action = null) {
        try {
            const body = { include_eval: debugMode };
            if (action) {
                body.action = action;
            }

            const resp = await fetch('/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });

            const data = await resp.json();
            if (!resp.ok || data.status === 'error') {
                throw new Error(data.message || 'Step execution failed');
            }

            currentState = data.state;
            renderState(currentState);

            const lastTrace = currentState.latest_trace;
            if (lastTrace) {
                const obsSummary = lastTrace.observations && lastTrace.observations.length > 0
                    ? lastTrace.observations.join('; ')
                    : lastTrace.rationale || '—';
                addLogEntry(
                    String(lastTrace.step).padStart(2, '0'),
                    lastTrace.action.toUpperCase(),
                    lastTrace.position,
                    lastTrace.battery,
                    lastTrace.remaining_time,
                    obsSummary
                );
            }

            const actionLabel = action || (lastTrace ? lastTrace.action : 'step');
            showFeedback(`Step ${currentState.step}: ${actionLabel} executed.`, 'success');

            if (currentState.status === 'completed') {
                stopPolling();
                isPlaying = false;
                playPauseBtn.textContent = 'Play';
                showFeedback(`Mission finished: ${currentState.outcome}`, 'info');
            }
        } catch (err) {
            showFeedback(`Step error: ${err.message}`, 'error');
            console.error('Step error:', err);
        }
    }

    async function togglePlayPause() {
        if (isPlaying) {
            // Pause
            try {
                const resp = await fetch('/pause', { method: 'POST' });
                const data = await resp.json();
                if (data.state) {
                    currentState = data.state;
                    renderState(currentState);
                }
            } catch (err) {
                console.error('Pause error:', err);
            }
            stopPolling();
            isPlaying = false;
            playPauseBtn.textContent = 'Play';
            showFeedback('Mission paused.', 'info');
        } else {
            // Play
            if (currentState && currentState.status === 'completed') {
                showFeedback('Mission already completed. Click Restart for a new run.', 'info');
                return;
            }

            try {
                const resp = await fetch('/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        include_eval: debugMode,
                        speed: 0.35
                    })
                });
                const data = await resp.json();
                if (data.state) {
                    currentState = data.state;
                    renderState(currentState);
                }
            } catch (err) {
                console.error('Run error:', err);
            }

            isPlaying = true;
            playPauseBtn.textContent = 'Pause';
            showFeedback('Autonomous mission running...', 'success');
            startPolling();
        }
    }

    function startPolling() {
        stopPolling();
        pollIntervalId = setInterval(async () => {
            try {
                const resp = await fetch(`/state?eval=${debugMode}`);
                if (!resp.ok) return;
                const state = await resp.json();
                if (state.status === 'no_mission') return;

                currentState = state;
                renderState(currentState);

                const lastTrace = currentState.latest_trace;
                if (lastTrace && (!currentState._lastLoggedStep || currentState._lastLoggedStep !== lastTrace.step)) {
                    currentState._lastLoggedStep = lastTrace.step;
                    const obsSummary = lastTrace.observations && lastTrace.observations.length > 0
                        ? lastTrace.observations.join('; ')
                        : lastTrace.rationale || '—';
                    addLogEntry(
                        String(lastTrace.step).padStart(2, '0'),
                        lastTrace.action.toUpperCase(),
                        lastTrace.position,
                        lastTrace.battery,
                        lastTrace.remaining_time,
                        obsSummary
                    );
                }

                if (state.status === 'completed') {
                    stopPolling();
                    isPlaying = false;
                    playPauseBtn.textContent = 'Play';
                    showFeedback(`Mission finished: ${state.outcome}`, 'info');
                }
            } catch (err) {
                console.error('Polling error:', err);
            }
        }, 300);
    }

    function stopPolling() {
        if (pollIntervalId) {
            clearInterval(pollIntervalId);
            pollIntervalId = null;
        }
    }

    // --- Rendering Logic ---

    function renderState(state) {
        if (!state) return;

        // 1. Live Stats
        agentMode.textContent = state.agent_mode || state.status || 'idle';
        if (state.target_tile) {
            targetTile.textContent = `(${state.target_tile[0]}, ${state.target_tile[1]})`;
        }
        distSafety.textContent = state.dist_safety !== undefined ? `${state.dist_safety} tiles` : '—';
        rechargeCount.textContent = state.recharge_count || 0;

        if (state.battery !== undefined) {
            const pct = state.battery_pct !== undefined ? state.battery_pct : state.battery;
            batteryLevel.textContent = `${pct.toFixed(1)}% (${state.battery.toFixed(1)} Wh)`;
            if (batteryFill) {
                batteryFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
                batteryFill.className = 'battery-fill';
                if (pct <= 20) {
                    batteryFill.classList.add('critical');
                } else if (pct <= 45) {
                    batteryFill.classList.add('warning');
                }
            }
        }

        if (state.time_left !== undefined) {
            timeLeft.textContent = `${state.time_left.toFixed(1)}s / ${state.total_time ? state.total_time.toFixed(0) + 's' : '—'}`;
        }

        // 2. Belief Summary
        if (state.map_estimate) {
            beliefLocation.textContent = `(${state.map_estimate[0]}, ${state.map_estimate[1]})`;
        }
        if (state.confidence !== undefined) {
            beliefConfidence.textContent = `${(state.confidence * 100).toFixed(2)}% (MAP)`;
        }
        if (state.entropy !== undefined) {
            beliefEntropy.textContent = `${state.entropy.toFixed(3)} nats`;
        }
        if (state.cells_known !== undefined && state.world) {
            const totalCells = state.world.width * state.world.height;
            const pct = ((state.cells_known / totalCells) * 100).toFixed(1);
            beliefCellsKnown.textContent = `${state.cells_known} / ${totalCells} (${pct}%)`;
        }

        // 3. Status Tags
        updateStatusTags(state);

        // 4. View Banner & Legend EVE Indicator
        if (debugMode) {
            viewInfo.textContent = '⚠️ EVALUATION ONLY — GROUND TRUTH REVEALED (BENCHMARKING)';
            viewInfo.classList.add('eval-active');
            if (legendEve) legendEve.style.display = 'inline-flex';
        } else {
            viewInfo.textContent = 'OPERATIONAL VIEW — STRICT INFORMATION BARRIER ACTIVE';
            viewInfo.classList.remove('eval-active');
            if (legendEve) legendEve.style.display = 'none';
        }

        // 5. Exploration Grid
        drawGrid(state);
    }

    function updateStatusTags(state) {
        statusTags.innerHTML = '';

        const addTag = (text, type = 'cyan') => {
            const span = document.createElement('span');
            span.className = `tag ${type}`;
            span.textContent = text;
            statusTags.appendChild(span);
        };

        if (state.outcome) {
            if (state.outcome.includes('located')) {
                addTag(`OUTCOME: ${state.outcome.toUpperCase()}`, 'pink');
            } else if (state.outcome.includes('exhausted')) {
                addTag(`OUTCOME: ${state.outcome.toUpperCase()}`, 'red');
            } else {
                addTag(`OUTCOME: ${state.outcome.toUpperCase()}`, 'amber');
            }
        } else if (state.status === 'running') {
            addTag('AUTONOMOUS ACTIVE', 'cyan');
        } else if (state.status === 'paused') {
            addTag('PAUSED', 'amber');
        } else {
            addTag('STANDBY', 'cyan');
        }

        if (state.battery_pct !== undefined && state.battery_pct <= 25) {
            addTag('BATTERY CRITICAL', 'red');
        }

        if (state.clues && state.clues.length > 0) {
            addTag(`CLUES: ${state.clues.length}`, 'amber');
        }
    }

    function drawGrid(state) {
        if (!state || !state.world) return;

        const w = state.world.width;
        const h = state.world.height;

        // Calculate dynamic cell size to fit viewport nicely
        const wrapper = gridContainer.parentElement;
        const availableW = wrapper.clientWidth - 40;
        const availableH = wrapper.clientHeight - 40;
        const cellSize = Math.max(16, Math.min(32, Math.floor(Math.min(availableW / w, availableH / h))));

        gridContainer.style.setProperty('--cell-size', `${cellSize}px`);
        gridContainer.style.setProperty('--grid-width', w);
        gridContainer.style.setProperty('--grid-height', h);

        // Precompute lookup sets for fast rendering
        const knownSet = new Set(state.known_cells || []);
        const obstacleSet = new Set(state.known_obstacles || []);
        const hazardSet = new Set(state.known_hazards || []);
        const stationSet = new Set(state.known_stations || []);
        const markerSet = new Set((state.pending_clue_markers || []).map(p => `${p[0]},${p[1]}`));
        const cluePickupSet = new Set((state.clues || []).filter(c => c.pickup).map(c => `${c.pickup[0]},${c.pickup[1]}`));
        const pathSet = new Set((state.path || []).map(p => `${p[0]},${p[1]}`));

        const robotPos = state.pos || [0, 0];
        const mapEst = state.map_estimate || [-1, -1];

        // Evaluation ground-truth sets (only active when debugMode is checked)
        const evalData = (debugMode && state.evaluation) ? state.evaluation : null;
        const evalObstacles = evalData ? new Set(evalData.obstacles.map(p => `${p[0]},${p[1]}`)) : null;
        const evalHazards = evalData ? new Set(evalData.hazards.map(p => `${p[0]},${p[1]}`)) : null;
        const evalStations = evalData ? new Set(evalData.stations.map(p => `${p[0]},${p[1]}`)) : null;
        const evalEvePos = evalData ? evalData.eve_position : null;

        // Find maximum probability for heatmap normalisation
        let maxProb = 0.0001;
        if (heatmapVisible && state.belief_map) {
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const val = state.belief_map[y] ? state.belief_map[y][x] : 0;
                    if (val > maxProb) maxProb = val;
                }
            }
        }

        // Rebuild or reuse grid cells
        const expectedCellCount = w * h;
        if (gridContainer.children.length !== expectedCellCount) {
            gridContainer.innerHTML = '';
            const fragment = document.createDocumentFragment();
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    cell.dataset.x = x;
                    cell.dataset.y = y;

                    const heatmapLayer = document.createElement('div');
                    heatmapLayer.className = 'heatmap-layer';
                    cell.appendChild(heatmapLayer);

                    cell.addEventListener('mouseenter', handleCellHover);
                    cell.addEventListener('mouseleave', hideCellTooltip);
                    fragment.appendChild(cell);
                }
            }
            gridContainer.appendChild(fragment);
        }

        // Update each cell's classes and overlays
        const cells = gridContainer.children;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = y * w + x;
                const cell = cells[idx];
                const key = `${x},${y}`;

                let classNames = 'cell';

                // Agent knowledge classes
                if (knownSet.has(key)) classNames += ' known';
                if (obstacleSet.has(key)) classNames += ' obstacle';
                if (hazardSet.has(key)) classNames += ' hazard';
                if (stationSet.has(key)) classNames += ' station';
                if (cluePickupSet.has(key)) classNames += ' clue';
                if (markerSet.has(key)) classNames += ' marker';
                if (pathSet.has(key)) classNames += ' path';

                // WALL-E position
                const isRobot = (x === robotPos[0] && y === robotPos[1]);
                if (isRobot) {
                    classNames += ' robot';
                }

                // MAP estimate
                if (x === mapEst[0] && y === mapEst[1]) {
                    classNames += ' map-estimate';
                }

                // Evaluation mode overlays
                if (evalData) {
                    if (evalEvePos && x === evalEvePos[0] && y === evalEvePos[1]) {
                        classNames += ' eval-eve';
                    }
                    if (evalObstacles && evalObstacles.has(key) && !obstacleSet.has(key)) {
                        classNames += ' eval-obstacle';
                    }
                    if (evalHazards && evalHazards.has(key) && !hazardSet.has(key)) {
                        classNames += ' eval-hazard';
                    }
                    if (evalStations && evalStations.has(key) && !stationSet.has(key)) {
                        classNames += ' eval-station';
                    }
                }

                cell.className = classNames;

                // Inner robot heading indicator
                let robotGlyph = cell.querySelector('.robot-heading');
                if (isRobot) {
                    const glyph = ORIENTATION_GLYPHS[state.orientation] || '▲';
                    if (!robotGlyph) {
                        robotGlyph = document.createElement('span');
                        robotGlyph.className = 'robot-heading';
                        cell.appendChild(robotGlyph);
                    }
                    robotGlyph.textContent = glyph;
                } else if (robotGlyph) {
                    robotGlyph.remove();
                }

                // Heatmap probability overlay
                const heatmapLayer = cell.querySelector('.heatmap-layer');
                if (heatmapLayer) {
                    if (heatmapVisible && state.belief_map && state.belief_map[y]) {
                        const prob = state.belief_map[y][x] || 0;
                        const norm = Math.min(1.0, prob / maxProb);
                        // Pink/Purple Sci-Fi Gradient
                        const alpha = Math.min(0.85, Math.pow(norm, 0.7) * 0.85);
                        if (alpha > 0.02) {
                            heatmapLayer.style.backgroundColor = `rgba(236, 72, 153, ${alpha.toFixed(3)})`;
                            heatmapLayer.style.boxShadow = alpha > 0.4 ? `inset 0 0 6px rgba(139, 92, 246, ${alpha.toFixed(3)})` : 'none';
                        } else {
                            heatmapLayer.style.backgroundColor = 'transparent';
                            heatmapLayer.style.boxShadow = 'none';
                        }
                    } else {
                        heatmapLayer.style.backgroundColor = 'transparent';
                        heatmapLayer.style.boxShadow = 'none';
                    }
                }
            }
        }
    }

    // --- Interactive Tooltip ---

    function handleCellHover(e) {
        if (!currentState) return;
        const x = parseInt(e.currentTarget.dataset.x, 10);
        const y = parseInt(e.currentTarget.dataset.y, 10);
        const key = `${x},${y}`;

        const isRobot = (currentState.pos && x === currentState.pos[0] && y === currentState.pos[1]);
        const isKnown = (currentState.known_cells || []).includes(key);
        const isObstacle = (currentState.known_obstacles || []).includes(key);
        const isHazard = (currentState.known_hazards || []).includes(key);
        const isStation = (currentState.known_stations || []).includes(key);
        const isMarker = (currentState.pending_clue_markers || []).some(m => m[0] === x && m[1] === y);
        const isClue = (currentState.clues || []).some(c => c.pickup && c.pickup[0] === x && c.pickup[1] === y);

        const lines = [];
        lines.push(`<div class="cell-tooltip-title">Tile Coordinates: (${x}, ${y})</div>`);

        if (isRobot) lines.push(`<span style="color: var(--amber-primary);">🤖 WALL-E Active [Facing ${currentState.orientation}]</span>`);
        if (isKnown) lines.push(`<span style="color: var(--cyan-primary);">✓ Explored Open Terrain</span>`);
        else lines.push(`<span style="color: var(--text-muted);">? Unexplored Sensor Fog</span>`);

        if (isObstacle) lines.push(`<span style="color: #cbd5e1;">⬛ Terrain Obstacle</span>`);
        if (isHazard) lines.push(`<span style="color: var(--hazard-red);">▲ Environmental Hazard</span>`);
        if (isStation) lines.push(`<span style="color: var(--station-green);">⚡ Solar Recharge Station</span>`);
        if (isClue) lines.push(`<span style="color: var(--amber-bright);">★ Collected Field Clue</span>`);
        if (isMarker) lines.push(`<span style="color: var(--amber-bright);">◇ Uncollected Clue Marker</span>`);

        if (currentState.belief_map && currentState.belief_map[y]) {
            const prob = currentState.belief_map[y][x] || 0;
            lines.push(`<span style="color: var(--pink-heatmap);">P(EVE): ${(prob * 100).toFixed(3)}%</span>`);
        }

        if (currentState.pos) {
            const dist = Math.abs(x - currentState.pos[0]) + Math.abs(y - currentState.pos[1]);
            lines.push(`<span style="color: var(--text-secondary);">Manhattan Dist to WALL-E: ${dist}</span>`);
        }

        // Ground truth inspection (evaluation view only)
        if (debugMode && currentState.evaluation) {
            const ev = currentState.evaluation;
            if (ev.eve_position && ev.eve_position[0] === x && ev.eve_position[1] === y) {
                lines.push(`<div style="margin-top: 4px; padding-top: 4px; border-top: 1px dashed #f43f5e; color: #f43f5e; font-weight: bold;">🎯 TRUE EVE LOCATION [Status: ${ev.eve_status}]</div>`);
            }
        }

        showCellTooltip(e, lines.join('<br>'));
    }

    function showCellTooltip(event, htmlContent) {
        hideCellTooltip();
        const tooltip = document.createElement('div');
        tooltip.className = 'cell-tooltip';
        tooltip.innerHTML = htmlContent;
        tooltip.style.left = `${event.pageX + 14}px`;
        tooltip.style.top = `${event.pageY + 14}px`;
        document.body.appendChild(tooltip);
        activeTooltip = tooltip;
    }

    function hideCellTooltip() {
        if (activeTooltip) {
            activeTooltip.remove();
            activeTooltip = null;
        }
    }

    // --- Action Trace Logging ---

    function addLogEntry(step, action, pos, battery, timeVal, notes) {
        if (!logContainer) return;

        // Clear initial placeholder if present
        const initial = logContainer.querySelector('.log-entry.initial');
        if (initial) initial.remove();

        const entry = document.createElement('div');
        entry.className = 'log-entry';

        const posStr = pos ? `(${pos[0]}, ${pos[1]})` : '—';
        const battStr = typeof battery === 'number' ? `${battery.toFixed(1)}%` : `${battery}`;
        const timeStr = typeof timeVal === 'number' ? `${timeVal.toFixed(1)}s` : `${timeVal}`;

        entry.innerHTML = `
            <div class="log-entry-header">
                <span>STEP ${step}</span>
                <span class="log-entry-action">${action}</span>
            </div>
            <div class="log-entry-details">
                <span>POS: ${posStr}</span>
                <span>BAT: ${battStr}</span>
                <span>TIME: ${timeStr}</span>
            </div>
            <div class="log-entry-rationale">${notes || ''}</div>
        `;

        logContainer.insertBefore(entry, logContainer.firstChild);

        // Limit stored log entries to 60
        while (logContainer.children.length > 60) {
            logContainer.removeChild(logContainer.lastChild);
        }
    }

    function showFeedback(msg, type = 'info') {
        if (!actionFeedback) return;
        actionFeedback.textContent = msg;
        actionFeedback.className = `action-feedback ${type}`;
    }

    // --- Event Listeners ---

    // Control buttons
    stepBtn.addEventListener('click', () => executeStep());
    playPauseBtn.addEventListener('click', () => togglePlayPause());
    restartBtn.addEventListener('click', () => resetMission());

    // Config buttons
    loadDefaultBtn.addEventListener('click', async () => {
        try {
            const resp = await fetch('/config/default');
            const data = await resp.json();
            if (data.yaml) {
                missionConfig.value = data.yaml;
                await resetMission();
            }
        } catch (err) {
            showFeedback(`Failed to load default config: ${err.message}`, 'error');
        }
    });

    newSeedBtn.addEventListener('click', async () => {
        const newSeed = Math.floor(Math.random() * 89999) + 10000;
        let val = missionConfig.value;
        if (/seed:\s*\d+/.test(val)) {
            val = val.replace(/seed:\s*\d+/, `seed: ${newSeed}`);
        } else if (/"seed":\s*\d+/.test(val)) {
            val = val.replace(/"seed":\s*\d+/, `"seed": ${newSeed}`);
        } else {
            val += `\n# override seed\nworld:\n  seed: ${newSeed}\n`;
        }
        missionConfig.value = val;
        showFeedback(`New seed ${newSeed} applied. Re-initializing mission...`, 'info');
        await resetMission();
    });

    clearLogBtn.addEventListener('click', () => {
        if (logContainer) {
            logContainer.innerHTML = '<div class="log-entry initial">Log cleared. Simulator active.</div>';
        }
    });

    // Heatmap toggle
    heatmapToggle.addEventListener('click', () => {
        heatmapVisible = !heatmapVisible;
        heatmapToggle.textContent = heatmapVisible ? 'Hide Heatmap' : 'Show Heatmap';
        if (currentState) {
            drawGrid(currentState);
        }
    });

    // Debug mode switch (ground truth)
    debugModeInput.addEventListener('change', async (e) => {
        debugMode = e.target.checked;
        if (debugMode) {
            showFeedback('Ground truth revealed for evaluation only. Information barrier preserved.', 'info');
        } else {
            showFeedback('Evaluation ground truth hidden.', 'info');
        }
        if (currentState) {
            try {
                const resp = await fetch(`/state?eval=${debugMode}`);
                if (resp.ok) {
                    currentState = await resp.json();
                }
            } catch (err) {
                console.error('State refresh error:', err);
            }
            renderState(currentState);
        }
    });

    // Manual action buttons
    document.querySelectorAll('.action-buttons .btn[data-action]').forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.getAttribute('data-action');
            if (action) {
                executeStep(action);
            }
        });
    });

    // Resize handler for responsive grid cell sizing
    window.addEventListener('resize', () => {
        if (currentState) {
            drawGrid(currentState);
        }
    });
});