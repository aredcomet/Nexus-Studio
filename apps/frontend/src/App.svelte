<script>
    import { onMount, tick } from 'svelte';
    import { marked } from 'marked';
    import DOMPurify from 'dompurify';
    
    // Custom marked renderer
    const renderer = {
        code(token) {
            const lang = token.lang;
            const text = token.text;
            return `<pre class="bg-slate-950/80 p-3.5 my-3 rounded-xl border border-slate-800/80 font-mono text-[13px] text-slate-200 overflow-x-auto select-text"><div class="text-[10px] text-slate-500 uppercase font-sans font-bold tracking-wider mb-1.5 border-b border-slate-800/40 pb-1">${lang || 'code'}</div><code>${DOMPurify.sanitize(text)}</code></pre>`;
        },
        codespan(token) {
            return `<code class="bg-[var(--bg-input)] border border-[var(--border-color)] px-1.5 py-0.5 rounded font-mono text-[13px] text-[var(--text-primary)]">${DOMPurify.sanitize(token.text)}</code>`;
        }
    };
    marked.use({ renderer });

    // Model Registry
    let PRESETS = $state({});

    // Server Config
    let serverPort = $state(8089);
    let isConnected = $state(false);
    let activeModel = $state('None');
    let activeType = $state('None');
    let activePaths = $state({});

    // Model Loader State
    let presetKey = $state('');
    let modelType = $state('recurrentgemma');
    let configPath = $state('');
    let weightsPath = $state('');
    let tokenizerPath = $state('');
    let adapterPath = $state('');
    let attnWindow = $state('');
    let chatTemplatePath = $state('');
    let ignoreLayers = $state([]);
    let draftModelPath = $state('');
    let draftKind = $state('');
    let draftBlockSize = $state('');

    // Hyperparameters
    let temperature = $state(0.7);
    let maxTokens = $state(256);
    let topP = $state(0.9);
    let topK = $state(50);
    let repeatPenalty = $state(1.1);
    let systemInstructions = $state('');

    // Chat Interface State
    let chatInput = $state('');
    let chatHistory = $state([]);
    let isGenerating = $state(false);
    let isModelLoaded = $state(false);

    // Loader Dialog State
    let showLoader = $state(false);
    let loaderTitle = $state('Loading Model...');
    let loaderSubtitle = $state('');

    // Inline Editing
    let editingIndex = $state(-1);
    let editBuffer = $state('');

    let messagesContainer = $state(null);
    let abortController = null;

    // Chat History LocalStorage States
    let sessions = $state([]);
    let folders = $state([]);
    let currentChatId = $state(null);
    let searchQuery = $state('');
    let showModelLoaderModal = $state(false);
    let isRightSidebarOpen = $state(true);
    let isLeftSidebarOpen = $state(true);
    let defaultReasoningCollapsed = $state(true);
    let hideReasoningBlocks = $state(false);
    let enableReasoning = $state(true);
    let expandedThoughts = $state({});
    let expandedToolArgs = $state({});
    let expandedToolResults = $state({});
    let theme = $state('dark');
    let activeMemGb = $state(0.0);
    let cacheMemGb = $state(0.0);
    let peakMemGb = $state(0.0);
    let gpuLimitGb = $state(0.0);
    let systemRamUsedGb = $state(null);
    let systemRamTotalGb = $state(null);
    let rightSidebarPane = $state('params');
    let toolCallingSupported = $state(false);
    let mcpServers = $state([]);
    let mcpSearchQuery = $state('');
    let showAddMcpModal = $state(false);
    let mcpExpandedServers = $state({});
    let newMcpName = $state('');
    let newMcpCommand = $state('');
    let newMcpArgs = $state('');
    let collapsedSections = $state({
        sampling: false,
        system: false,
        reasoning: false,
        port: true
    });

    function generateId() {
        return Math.random().toString(36).substring(2, 11);
    }

    function loadFromLocalStorage() {
        const storedSessions = localStorage.getItem('mlx_gateway_sessions');
        const storedFolders = localStorage.getItem('mlx_gateway_folders');
        
        if (storedSessions) {
            sessions = JSON.parse(storedSessions);
            let mutated = false;
            sessions.forEach(s => {
                if (!s.createdAt) {
                    s.createdAt = Date.now();
                    mutated = true;
                }
            });
            if (mutated) saveToLocalStorage();
        } else {
            const defaultId = generateId();
            sessions = [{
                id: defaultId,
                title: 'New Conversation',
                folderId: null,
                createdAt: Date.now(),
                systemInstructions: '',
                temperature: 0.7,
                maxTokens: 256,
                topP: 0.9,
                topK: 50,
                repeatPenalty: 1.1,
                history: []
            }];
            saveToLocalStorage();
        }
        
        if (storedFolders) {
            folders = JSON.parse(storedFolders);
            let mutated = false;
            folders.forEach(f => {
                if (f.parentId === undefined) {
                    f.parentId = null;
                    mutated = true;
                }
            });
            if (mutated) saveToLocalStorage();
        } else {
            folders = [];
        }

        const storedHideReasoning = localStorage.getItem('mlx_gateway_hide_reasoning');
        if (storedHideReasoning !== null) {
            hideReasoningBlocks = JSON.parse(storedHideReasoning);
        }
        const storedDefaultCollapsed = localStorage.getItem('mlx_gateway_default_collapsed');
        if (storedDefaultCollapsed !== null) {
            defaultReasoningCollapsed = JSON.parse(storedDefaultCollapsed);
        }
        const storedEnableReasoning = localStorage.getItem('mlx_gateway_enable_reasoning');
        if (storedEnableReasoning !== null) {
            enableReasoning = JSON.parse(storedEnableReasoning);
        }
        const storedTheme = localStorage.getItem('mlx_gateway_theme');
        if (storedTheme !== null) {
            theme = storedTheme;
        }
        
        if (sessions.length > 0) {
            loadSession(sessions[0].id);
        }
    }
    
    function saveToLocalStorage() {
        localStorage.setItem('mlx_gateway_sessions', JSON.stringify(sessions));
        localStorage.setItem('mlx_gateway_folders', JSON.stringify(folders));
        localStorage.setItem('mlx_gateway_hide_reasoning', JSON.stringify(hideReasoningBlocks));
        localStorage.setItem('mlx_gateway_default_collapsed', JSON.stringify(defaultReasoningCollapsed));
        localStorage.setItem('mlx_gateway_enable_reasoning', JSON.stringify(enableReasoning));
        localStorage.setItem('mlx_gateway_theme', theme);
    }

    function loadSession(id) {
        const session = sessions.find(s => s.id === id);
        if (session) {
            currentChatId = id;
            chatHistory = session.history || [];
            systemInstructions = session.systemInstructions || '';
            temperature = session.temperature !== undefined ? session.temperature : 0.7;
            maxTokens = session.maxTokens !== undefined ? session.maxTokens : 256;
            topP = session.topP !== undefined ? session.topP : 0.9;
            topK = session.topK !== undefined ? session.topK : 50;
            repeatPenalty = session.repeatPenalty !== undefined ? session.repeatPenalty : 1.1;
            chatInput = '';
        }
    }

    function createNewChat(folderId = null) {
        const newId = generateId();
        const newSession = {
            id: newId,
            title: 'New Conversation',
            folderId: folderId,
            createdAt: Date.now(),
            systemInstructions: systemInstructions,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            topK: topK,
            repeatPenalty: repeatPenalty,
            history: []
        };
        sessions = [newSession, ...sessions];
        currentChatId = newId;
        chatHistory = [];
        chatInput = '';
        saveToLocalStorage();
    }

    function deleteSession(id, e) {
        if (e) e.stopPropagation();
        if (!confirm("Are you sure you want to delete this chat?")) return;
        
        sessions = sessions.filter(s => s.id !== id);
        saveToLocalStorage();
        
        if (currentChatId === id) {
            if (sessions.length > 0) {
                loadSession(sessions[0].id);
            } else {
                createNewChat();
            }
        }
    }

    function renameSession(id, e) {
        if (e) e.stopPropagation();
        const session = sessions.find(s => s.id === id);
        if (session) {
            const newTitle = prompt("Enter new chat title:", session.title);
            if (newTitle && newTitle.trim()) {
                session.title = newTitle.trim();
                saveToLocalStorage();
            }
        }
    }

    // Folders Actions
    function createFolder(parentId = null, e = null) {
        if (e) e.stopPropagation();
        const name = prompt("Enter folder name:");
        if (name && name.trim()) {
            const newFolder = {
                id: generateId(),
                name: name.trim(),
                parentId: parentId,
                isExpanded: true
            };
            folders = [...folders, newFolder];
            if (parentId) {
                const parent = folders.find(f => f.id === parentId);
                if (parent) parent.isExpanded = true;
            }
            saveToLocalStorage();
        }
    }

    function renameFolder(id, e) {
        if (e) e.stopPropagation();
        const folder = folders.find(f => f.id === id);
        if (folder) {
            const newName = prompt("Enter new folder name:", folder.name);
            if (newName && newName.trim()) {
                folder.name = newName.trim();
                saveToLocalStorage();
            }
        }
    }

    function isDescendant(parentCheckId, childCheckId) {
        if (!childCheckId) return false;
        const folder = folders.find(f => f.id === childCheckId);
        if (!folder) return false;
        if (folder.parentId === parentCheckId) return true;
        return isDescendant(parentCheckId, folder.parentId);
    }

    function getFolderPath(folderId) {
        const folder = folders.find(f => f.id === folderId);
        if (!folder) return '';
        if (!folder.parentId) return folder.name;
        return `${getFolderPath(folder.parentId)} / ${folder.name}`;
    }

    function folderHasMatches(folderId) {
        const query = searchQuery.toLowerCase();
        if (!query) return true;
        
        // Check if any chat in this folder matches
        const hasMatchingChat = sessions.some(s => s.folderId === folderId && s.title.toLowerCase().includes(query));
        if (hasMatchingChat) return true;
        
        // Check if any subfolder matches
        const subfolders = folders.filter(f => f.parentId === folderId);
        return subfolders.some(sf => folderHasMatches(sf.id));
    }

    function moveFolder(id, e) {
        if (e) e.stopPropagation();
        const folder = folders.find(f => f.id === id);
        if (!folder) return;
        
        const validFolders = folders.filter(f => f.id !== id && !isDescendant(id, f.id));
        
        let folderOptions = `Move folder "${folder.name}" to another folder.\nEnter folder name, or leave blank for Root:\n\nAvailable Folders:\n`;
        validFolders.forEach(f => {
            folderOptions += `- ${getFolderPath(f.id)}\n`;
        });
        
        const targetName = prompt(folderOptions);
        if (targetName === null) return;
        
        if (targetName.trim() === '') {
            folder.parentId = null;
        } else {
            const target = validFolders.find(f => f.name.toLowerCase() === targetName.trim().toLowerCase() || getFolderPath(f.id).toLowerCase() === targetName.trim().toLowerCase());
            if (target) {
                folder.parentId = target.id;
                target.isExpanded = true;
            } else {
                alert("Invalid folder name. Folder not moved.");
                return;
            }
        }
        saveToLocalStorage();
    }

    function deleteFolder(id, e) {
        if (e) e.stopPropagation();
        const folder = folders.find(f => f.id === id);
        if (!folder) return;
        if (!confirm(`Are you sure you want to delete folder "${folder.name}"? (Contents will be moved to its parent directory)`)) return;
        
        const pId = folder.parentId;
        
        sessions.forEach(s => {
            if (s.folderId === id) {
                s.folderId = pId;
            }
        });
        
        folders.forEach(f => {
            if (f.parentId === id) {
                f.parentId = pId;
            }
        });
        
        folders = folders.filter(f => f.id !== id);
        saveToLocalStorage();
    }

    function moveSessionToFolder(id, e) {
        if (e) e.stopPropagation();
        const session = sessions.find(s => s.id === id);
        if (!session) return;
        
        let folderOptions = `Move chat "${session.title}" to a folder.\nEnter folder name, or leave blank for Root:\n\nAvailable Folders:\n`;
        folders.forEach(f => {
            folderOptions += `- ${getFolderPath(f.id)}\n`;
        });
        
        const targetName = prompt(folderOptions);
        if (targetName === null) return;
        
        if (targetName.trim() === '') {
            session.folderId = null;
        } else {
            const target = folders.find(f => f.name.toLowerCase() === targetName.trim().toLowerCase() || getFolderPath(f.id).toLowerCase() === targetName.trim().toLowerCase());
            if (target) {
                session.folderId = target.id;
                target.isExpanded = true;
            } else {
                if (confirm(`Folder "${targetName.trim()}" does not exist. Create it?`)) {
                    const newFolderId = generateId();
                    const newFolder = {
                        id: newFolderId,
                        name: targetName.trim(),
                        parentId: null,
                        isExpanded: true
                    };
                    folders = [...folders, newFolder];
                    session.folderId = newFolderId;
                }
            }
        }
        saveToLocalStorage();
    }



    // Auto-save effect
    $effect(() => {
        if (currentChatId) {
            const idx = sessions.findIndex(s => s.id === currentChatId);
            if (idx !== -1) {
                sessions[idx].history = chatHistory;
                sessions[idx].systemInstructions = systemInstructions;
                sessions[idx].temperature = temperature;
                sessions[idx].maxTokens = maxTokens;
                sessions[idx].topP = topP;
                sessions[idx].topK = topK;
                sessions[idx].repeatPenalty = repeatPenalty;
                
                if (sessions[idx].title === 'New Conversation' && chatHistory.length > 0) {
                    const firstUserMsg = chatHistory.find(m => m.role === 'user');
                    if (firstUserMsg) {
                        const content = firstUserMsg.content;
                        sessions[idx].title = content.length > 24 ? content.substring(0, 24) + '...' : content;
                    }
                }
                
                saveToLocalStorage();
            }
        }
    });

    function stopGeneration() {
        if (abortController) {
            abortController.abort();
        }
    }

    // Svelte action for auto-resizing textareas
    function autoresize(node) {
        function resize() {
            node.style.height = 'auto';
            const maxHeight = window.innerHeight * 0.4; // 40vh
            if (node.scrollHeight > maxHeight) {
                node.style.height = `${maxHeight}px`;
                node.style.overflowY = 'auto';
            } else {
                node.style.height = `${node.scrollHeight}px`;
                node.style.overflowY = 'hidden';
            }
        }
        
        node.addEventListener('input', resize);
        
        // Wait for bindings to complete, then resize
        setTimeout(resize, 0);
        
        return {
            update() {
                resize();
            },
            destroy() {
                node.removeEventListener('input', resize);
            }
        };
    }

    // Helpers
    function getServerUrl() {
        return `http://127.0.0.1:${serverPort}`;
    }

    async function checkServerStatus() {
        try {
            const res = await fetch(`${getServerUrl()}/v1/model/status`);
            if (res.ok) {
                const data = await res.json();
                isConnected = true;
                isModelLoaded = data.loaded;
                activeModel = data.model_name || 'None';
                activeType = data.model_type || 'None';
                activePaths = data.paths || {};
                activeMemGb = data.active_mem_gb || 0.0;
                cacheMemGb = data.cache_mem_gb || 0.0;
                peakMemGb = data.peak_mem_gb || 0.0;
                gpuLimitGb = data.gpu_limit_gb || 0.0;
                systemRamUsedGb = data.system_ram_used_gb !== undefined ? data.system_ram_used_gb : null;
                systemRamTotalGb = data.system_ram_total_gb !== undefined ? data.system_ram_total_gb : null;
                toolCallingSupported = data.tool_calling_supported || false;
            } else {
                isConnected = false;
                isModelLoaded = false;
                activeModel = 'None';
                activeType = 'None';
                activeMemGb = 0.0;
                cacheMemGb = 0.0;
                peakMemGb = 0.0;
                gpuLimitGb = 0.0;
                systemRamUsedGb = null;
                systemRamTotalGb = null;
                toolCallingSupported = false;
            }
        } catch (e) {
            isConnected = false;
            isModelLoaded = false;
            activeModel = 'None';
            activeType = 'None';
            activeMemGb = 0.0;
            cacheMemGb = 0.0;
            peakMemGb = 0.0;
            gpuLimitGb = 0.0;
            systemRamUsedGb = null;
            systemRamTotalGb = null;
            toolCallingSupported = false;
        }
    }

    let filteredMcpServers = $derived.by(() => {
        const query = mcpSearchQuery.toLowerCase().trim();
        if (!query) return mcpServers;
        return mcpServers.filter(s => s.name.toLowerCase().includes(query));
    });

    async function loadMcpServers() {
        try {
            const res = await fetch(`${getServerUrl()}/v1/mcp/servers`);
            if (res.ok) {
                mcpServers = await res.json();
            }
        } catch (e) {
            console.error("Error loading MCP servers:", e);
        }
    }

    async function toggleMcpServer(name, enabled) {
        try {
            const res = await fetch(`${getServerUrl()}/v1/mcp/servers/toggle`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, enabled })
            });
            if (res.ok) {
                await loadMcpServers();
            }
        } catch (e) {
            console.error("Error toggling MCP server:", e);
        }
    }

    async function toggleMcpTool(serverName, toolName, enabled) {
        try {
            const res = await fetch(`${getServerUrl()}/v1/mcp/servers/tools/toggle`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ server_name: serverName, tool_name: toolName, enabled })
            });
            if (res.ok) {
                await loadMcpServers();
            }
        } catch (e) {
            console.error("Error toggling MCP tool:", e);
        }
    }

    async function toggleMcpToolPermission(serverName, toolName, permission) {
        try {
            const res = await fetch(`${getServerUrl()}/v1/mcp/servers/tools/permission`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    server_name: serverName,
                    tool_name: toolName,
                    permission: permission
                })
            });
            if (res.ok) {
                await loadMcpServers();
            }
        } catch (e) {
            console.error("Error toggling tool permission:", e);
        }
    }

    async function handleApprovalDecision(index, approvalId, decision) {
        try {
            // Update UI state immediately
            chatHistory[index].pendingApproval = null;
            chatHistory[index].approvedStatus = decision;
            
            // Send decision to backend
            const res = await fetch(`${getServerUrl()}/v1/mcp/approve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    approval_id: approvalId,
                    decision: decision
                })
            });
            if (!res.ok) {
                console.error("Failed to approve tool call on server");
            }
        } catch (e) {
            console.error("Error approving tool call:", e);
        }
    }

    function toggleServerExpanded(name) {
        mcpExpandedServers[name] = !mcpExpandedServers[name];
    }

    async function refreshMcpTools(name) {
        await loadMcpServers();
    }

    async function installMcpServer() {
        if (!newMcpName.trim() || !newMcpCommand.trim()) {
            alert("Server ID and Command are required.");
            return;
        }
        const args = newMcpArgs.split(/[\n,]/)
            .map(a => a.trim())
            .filter(a => a.length > 0);
        try {
            const res = await fetch(`${getServerUrl()}/v1/mcp/install`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: newMcpName.trim(),
                    command: newMcpCommand.trim(),
                    args: args
                })
            });
            if (res.ok) {
                showAddMcpModal = false;
                newMcpName = '';
                newMcpCommand = '';
                newMcpArgs = '';
                await loadMcpServers();
            } else {
                const err = await res.text();
                alert(`Error installing: ${err}`);
            }
        } catch (e) {
            console.error("Error installing MCP server:", e);
        }
    }

    $effect(() => {
        if (rightSidebarPane === 'mcp') {
            loadMcpServers();
        }
    });

    // Chat Presets
    let CHAT_PRESETS = $state([]);
    let selectedChatPreset = $state('');

    async function loadChatPresets() {
        try {
            const res = await fetch(`${getServerUrl()}/v1/presets`);
            if (res.ok) {
                CHAT_PRESETS = await res.json();
                const def = CHAT_PRESETS.find(p => p.preset_name.toLowerCase() === 'default');
                if (def && !selectedChatPreset) {
                    selectedChatPreset = def.preset_name;
                    applyChatPreset(def);
                }
            }
        } catch (e) {
            console.error("Error loading chat presets:", e);
        }
    }

    function applyChatPreset(preset) {
        if (!preset) return;
        if (preset.sampling_params) {
            temperature = preset.sampling_params.temperature ?? temperature;
            topP = preset.sampling_params.top_p ?? topP;
            topK = preset.sampling_params.top_k ?? topK;
            repeatPenalty = preset.sampling_params.repetition_penalty ?? repeatPenalty;
            maxTokens = preset.sampling_params.max_tokens ?? maxTokens;
        }
        if (preset.reasoning_settings) {
            enableReasoning = preset.reasoning_settings.enable_reasoning ?? enableReasoning;
            defaultReasoningCollapsed = preset.reasoning_settings.collapse_by_default ?? defaultReasoningCollapsed;
            hideReasoningBlocks = preset.reasoning_settings.hide_reasoning ?? hideReasoningBlocks;
        }
        if (preset.system_instruction !== undefined) {
            systemInstructions = preset.system_instruction;
        }
    }

    function handleChatPresetChange(e) {
        const val = e.target.value;
        selectedChatPreset = val;
        if (val) {
            const p = CHAT_PRESETS.find(x => x.preset_name === val);
            if (p) applyChatPreset(p);
        }
    }

    async function saveChatPreset() {
        if (!selectedChatPreset) {
            alert("Please select a preset to override, or use 'Create New'.");
            return;
        }
        const payload = {
            preset_name: selectedChatPreset,
            sampling_params: {
                temperature,
                top_p: topP,
                top_k: topK,
                repetition_penalty: repeatPenalty,
                max_tokens: maxTokens
            },
            reasoning_settings: {
                enable_reasoning: enableReasoning,
                collapse_by_default: defaultReasoningCollapsed,
                hide_reasoning: hideReasoningBlocks
            },
            system_instruction: systemInstructions
        };

        try {
            const res = await fetch(`${getServerUrl()}/v1/presets/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                await loadChatPresets();
            }
        } catch (e) {
            console.error("Error saving preset:", e);
            alert("Error saving preset");
        }
    }

    async function saveAsNewPreset() {
        const name = prompt("Enter a name for this new preset:");
        if (!name) return;
        
        const payload = {
            preset_name: name,
            sampling_params: {
                temperature,
                top_p: topP,
                top_k: topK,
                repetition_penalty: repeatPenalty,
                max_tokens: maxTokens
            },
            reasoning_settings: {
                enable_reasoning: enableReasoning,
                collapse_by_default: defaultReasoningCollapsed,
                hide_reasoning: hideReasoningBlocks
            },
            system_instruction: systemInstructions
        };

        try {
            const res = await fetch(`${getServerUrl()}/v1/presets/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                await loadChatPresets();
                selectedChatPreset = name;
            }
        } catch (e) {
            console.error("Error saving preset:", e);
            alert("Error saving preset");
        }
    }

    async function loadModelsRegistry() {
        try {
            const res = await fetch(`${getServerUrl()}/v1/models/registry`);
            if (res.ok) {
                PRESETS = await res.json();
            }
        } catch (e) {
            console.error("Error loading models registry:", e);
        }
    }

    onMount(() => {
        loadModelsRegistry();
        loadChatPresets();
        checkServerStatus();
        loadFromLocalStorage();
        const interval = setInterval(checkServerStatus, 4000);
        return () => clearInterval(interval);
    });

    // Preset Selection updates inputs
    function handlePresetChange(e) {
        presetKey = e.target.value;
        if (presetKey && PRESETS[presetKey]) {
            const preset = PRESETS[presetKey];
            modelType = preset.engine || preset.type || 'standard';
            configPath = preset.config_path || preset.config || '';
            weightsPath = preset.weights_path || preset.weights || '';
            tokenizerPath = preset.tokenizer_path || preset.tokenizer || '';
            chatTemplatePath = preset.chat_template_path || preset.chatTemplate || '';
            ignoreLayers = preset.ignore_layers || [];
            draftModelPath = preset.draft_model_path || '';
            draftKind = preset.draft_kind || '';
            draftBlockSize = preset.draft_block_size !== undefined ? String(preset.draft_block_size) : '';
            adapterPath = '';
            attnWindow = '';
        }
    }

    // Load Model Endpoint call
    async function loadModel() {
        if (!isConnected) return;
        loaderTitle = "Loading Model...";
        loaderSubtitle = "Binding weights on your local Apple Silicon GPU VRAM. Please wait.";
        showLoader = true;

        const payload = {
            model_type: modelType,
            config_path: configPath.trim(),
            weights_path: weightsPath.trim(),
            tokenizer_path: tokenizerPath.trim(),
            chat_template_path: chatTemplatePath.trim() || null,
            adapter_path: adapterPath.trim() || null,
            attention_window: attnWindow.trim() ? parseInt(attnWindow) : null,
            ignore_layers: ignoreLayers.length > 0 ? ignoreLayers : null,
            draft_model_path: draftModelPath.trim() || null,
            draft_kind: draftKind.trim() || null,
            draft_block_size: draftBlockSize.trim() ? parseInt(draftBlockSize) : null
        };

        try {
            const res = await fetch(`${getServerUrl()}/v1/model/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (res.ok) {
                alert(`Successfully loaded: ${data.model_name} (${data.load_time_seconds.toFixed(2)}s)`);
            } else {
                throw new Error(data.detail || 'Loading failed');
            }
        } catch (e) {
            alert(`Error loading model: ${e.message}`);
        } finally {
            showLoader = false;
            checkServerStatus();
        }
    }

    // Unload Model Endpoint call
    async function unloadModel() {
        if (!isConnected) return;
        loaderTitle = "Unloading Model...";
        loaderSubtitle = "Reclaiming memory resources and clearing Metal VRAM cache.";
        showLoader = true;

        try {
            const res = await fetch(`${getServerUrl()}/v1/model/unload`, { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                alert(data.message);
            } else {
                throw new Error('Unload request failed');
            }
        } catch (e) {
            alert(`Error: ${e.message}`);
        } finally {
            showLoader = false;
            checkServerStatus();
        }
    }

    // Auto-Scroll to bottom
    function scrollToBottom() {
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    $effect(() => {
        if (chatHistory.length || isGenerating) {
            setTimeout(scrollToBottom, 30);
        }
    });

    // Chat Generation Streaming
    async function generateResponse(existingPlaceholder = null) {
        if (isGenerating || !isModelLoaded) return;
        isGenerating = true;

        abortController = new AbortController();
        const { signal } = abortController;

        // Add placeholder bubble for the assistant's streaming response
        if (existingPlaceholder) {
            chatHistory = [...chatHistory, existingPlaceholder];
        } else {
            let placeholderMsg = { role: 'assistant', content: '', isPlaceholder: true };
            chatHistory = [...chatHistory, placeholderMsg];
        }

        const requestMessages = [];
        if (systemInstructions.trim()) {
            requestMessages.push({ role: 'system', content: systemInstructions.trim() });
        }
        // Filter out the last empty assistant message
        chatHistory.slice(0, -1).forEach(h => {
            requestMessages.push({ role: h.role, content: h.content });
        });

        try {
            const url = `${getServerUrl()}/v1/chat/completions`;
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: requestMessages,
                    temperature: parseFloat(temperature),
                    max_tokens: parseInt(maxTokens),
                    top_p: parseFloat(topP),
                    top_k: parseInt(topK),
                    repeat_penalty: parseFloat(repeatPenalty),
                    stream: true,
                    enable_thinking: enableReasoning
                }),
                signal
            });

            if (!response.ok) {
                throw new Error('Server failed during response generation.');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let assistantText = '';
            let started = false;
            
            let startTime = performance.now();
            let firstTokenTime = 0;
            let tokensReceived = 0;
            let stopReason = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    const cleaned = line.trim();
                    if (cleaned === 'data: [DONE]') break;
                    
                    if (cleaned.startsWith('data: ')) {
                        try {
                            const parsed = JSON.parse(cleaned.substring(6));
                            
                            if (parsed.choices[0].finish_reason) {
                                let fr = parsed.choices[0].finish_reason;
                                stopReason = fr === 'stop' ? 'EOS Token Found' : (fr === 'length' ? 'Max Tokens Reached' : fr);
                            }
                            
                            const delta = parsed.choices[0].delta.content;
                            if (delta) {
                                if (!started) {
                                    started = true;
                                    firstTokenTime = performance.now() - startTime;
                                }
                                tokensReceived++;
                                assistantText += delta;
                                
                                let currentMsg = chatHistory[chatHistory.length - 1];
                                let newAlternatives = currentMsg.alternatives ? [...currentMsg.alternatives] : undefined;
                                if (newAlternatives) {
                                    newAlternatives[currentMsg.altIndex || 0] = assistantText;
                                }

                                // Update placeholder message in history reactively
                                chatHistory[chatHistory.length - 1] = {
                                    ...currentMsg,
                                    role: 'assistant',
                                    content: assistantText,
                                    isPlaceholder: false,
                                    alternatives: newAlternatives
                                };
                            }
                            
                            const toolApproval = parsed.choices[0].delta.tool_approval_required;
                            if (toolApproval) {
                                if (!started) {
                                    started = true;
                                }
                                let currentMsg = chatHistory[chatHistory.length - 1];
                                chatHistory[chatHistory.length - 1] = {
                                    ...currentMsg,
                                    role: 'assistant',
                                    content: assistantText,
                                    isPlaceholder: false,
                                    pendingApproval: toolApproval
                                };
                            }
                        } catch (e) {}
                    }
                }
            }
            
            if (!started) {
                chatHistory[chatHistory.length - 1] = {
                    role: 'assistant',
                    content: 'No tokens were generated. Check local server terminal.',
                    isPlaceholder: false
                };
            }
            
            // Finalize stats
            let totalTime = performance.now() - startTime;
            let generationTime = totalTime; // total time including TTFT
            // Use generationTime for tok/sec. If no tokens, avoid NaN.
            let tokPerSec = tokensReceived > 0 && generationTime > 0 ? (tokensReceived / (generationTime / 1000)).toFixed(2) : 0;
            
            let finalMsg = chatHistory[chatHistory.length - 1];
            chatHistory[chatHistory.length - 1] = {
                ...finalMsg,
                stats: {
                    tokens: tokensReceived,
                    ttft: firstTokenTime > 0 ? (firstTokenTime / 1000).toFixed(2) + 's' : 'N/A',
                    tokPerSec: tokPerSec,
                    stopReason: stopReason || 'Unknown'
                }
            };

        } catch (err) {
            if (err.name === 'AbortError') {
                console.log("Generation aborted by the user.");
                if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].role === 'assistant') {
                    chatHistory[chatHistory.length - 1] = {
                        role: 'assistant',
                        content: chatHistory[chatHistory.length - 1].content + " ⏹️ [Stopped]",
                        isPlaceholder: false
                    };
                }
            } else {
                console.error(err);
                chatHistory[chatHistory.length - 1] = {
                    role: 'assistant',
                    content: `Error: ${err.message || 'Could not connect to model server.'}`,
                    isPlaceholder: false,
                    isError: true
                };
            }
        } finally {
            isGenerating = false;
            abortController = null;
            // Focus main chat input again
            const input = document.getElementById('chat-textarea');
            if (input) input.focus();
        }
    }

    async function sendMessage() {
        const query = chatInput.trim();
        if (!query || isGenerating || !isModelLoaded) return;

        chatInput = '';
        chatHistory = [...chatHistory, { role: 'user', content: query }];
        await generateResponse();
    }

    function clearChats() {
        chatHistory = [];
    }

    // Edit User Message
    function startEditMessage(index) {
        editingIndex = index;
        editBuffer = chatHistory[index].content;
    }

    function cancelEdit() {
        editingIndex = -1;
        editBuffer = '';
    }

    async function saveEdit(index) {
        const text = editBuffer.trim();
        if (!text) return;
        
        let msg = chatHistory[index];
        if (msg.role === 'assistant') {
            msg.content = text;
            if (msg.alternatives) {
                msg.alternatives[msg.altIndex || 0] = text;
            }
            editingIndex = -1;
            editBuffer = '';
            saveToLocalStorage();
            return;
        }

        // Discard history from this message onward and update it
        chatHistory = chatHistory.slice(0, index);
        chatHistory = [...chatHistory, { role: 'user', content: text }];

        editingIndex = -1;
        editBuffer = '';

        await generateResponse();
    }

    // Retry Assistant Response
    async function retryMessage(index) {
        if (isGenerating || !isModelLoaded) return;
        
        let msgToRetry = chatHistory[index];
        let alts = msgToRetry.alternatives || [msgToRetry.content];
        
        // Slice off history starting from this assistant response
        chatHistory = chatHistory.slice(0, index);
        
        let placeholder = {
            role: 'assistant',
            content: '',
            isPlaceholder: true,
            alternatives: [...alts, ''],
            altIndex: alts.length
        };
        
        await generateResponse(placeholder);
    }

    function switchAlternative(index, step) {
        if (isGenerating) return;
        let msg = chatHistory[index];
        if (!msg.alternatives || msg.alternatives.length <= 1) return;
        
        let newIndex = (msg.altIndex || 0) + step;
        if (newIndex >= 0 && newIndex < msg.alternatives.length) {
            msg.altIndex = newIndex;
            msg.content = msg.alternatives[newIndex];
            
            // Truncate subsequent history to avoid inconsistent contexts
            chatHistory = chatHistory.slice(0, index + 1);
            saveToLocalStorage();
        }
    }

    function branchChat(index) {
        if (isGenerating) return;
        const branchHistory = JSON.parse(JSON.stringify(chatHistory.slice(0, index + 1)));
        const newId = Date.now().toString();
        const currentSession = sessions.find(s => s.id === currentChatId);
        const title = currentSession ? 'Branch of ' + currentSession.title : 'Branched Chat';
        
        sessions = [{ id: newId, title: title, history: branchHistory, updatedAt: Date.now() }, ...sessions];
        currentChatId = newId;
        chatHistory = branchHistory;
        saveToLocalStorage();
    }

    function deleteMessage(index) {
        if (isGenerating) return;
        if (confirm("Delete this message and all subsequent messages?")) {
            chatHistory = chatHistory.slice(0, index);
            saveToLocalStorage();
        }
    }

    async function continueMessage(index) {
        if (isGenerating || !isModelLoaded) return;
        isGenerating = true;
        abortController = new AbortController();
        const { signal } = abortController;
        
        // Ensure index is the last message
        chatHistory = chatHistory.slice(0, index + 1);
        
        const requestMessages = [];
        if (systemInstructions.trim()) requestMessages.push({ role: 'system', content: systemInstructions.trim() });
        chatHistory.forEach(h => requestMessages.push({ role: h.role, content: h.content }));
        
        // We will append to the current assistant message
        let assistantText = chatHistory[index].content || '';
        let currentMsg = chatHistory[index];
        let newAlternatives = currentMsg.alternatives ? [...currentMsg.alternatives] : undefined;

        try {
            const url = `${getServerUrl()}/v1/chat/completions`;
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: requestMessages,
                    temperature: parseFloat(temperature),
                    max_tokens: parseInt(maxTokens),
                    top_p: parseFloat(topP),
                    top_k: parseInt(topK),
                    repeat_penalty: parseFloat(repeatPenalty),
                    stream: true,
                    enable_thinking: enableReasoning
                }),
                signal
            });

            if (!response.ok) throw new Error('Server failed during response generation.');

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            
            let startTime = performance.now();
            let firstTokenTime = 0;
            let tokensReceived = 0;
            let stopReason = '';
            let started = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    const cleaned = line.trim();
                    if (cleaned === 'data: [DONE]') break;
                    if (cleaned.startsWith('data: ')) {
                        try {
                            const parsed = JSON.parse(cleaned.substring(6));
                            
                            if (parsed.choices[0].finish_reason) {
                                let fr = parsed.choices[0].finish_reason;
                                stopReason = fr === 'stop' ? 'EOS Token Found' : (fr === 'length' ? 'Max Tokens Reached' : fr);
                            }
                            
                            const delta = parsed.choices[0].delta.content;
                            if (delta) {
                                if (!started) {
                                    started = true;
                                    firstTokenTime = performance.now() - startTime;
                                }
                                tokensReceived++;
                                assistantText += delta;
                                if (newAlternatives) newAlternatives[currentMsg.altIndex || 0] = assistantText;
                                chatHistory[index] = {
                                    ...currentMsg,
                                    content: assistantText,
                                    alternatives: newAlternatives
                                };
                            }
                        } catch (e) {}
                    }
                }
            }
            
            // Finalize stats
            let totalTime = performance.now() - startTime;
            let generationTime = totalTime;
            let tokPerSec = tokensReceived > 0 && generationTime > 0 ? (tokensReceived / (generationTime / 1000)).toFixed(2) : 0;
            
            // Merge with existing stats if present
            let existingStats = currentMsg.stats || { tokens: 0, ttft: 'N/A' };
            let totalTokens = (existingStats.tokens || 0) + tokensReceived;
            
            chatHistory[index] = {
                ...chatHistory[index],
                stats: {
                    tokens: totalTokens,
                    ttft: firstTokenTime > 0 ? (firstTokenTime / 1000).toFixed(2) + 's' : existingStats.ttft,
                    tokPerSec: tokPerSec,
                    stopReason: stopReason || 'Unknown'
                }
            };
        } catch (err) {
            console.error(err);
        } finally {
            isGenerating = false;
            abortController = null;
            saveToLocalStorage();
        }
    }

    // Keydown handlers
    function handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }

    // Parse Google Gemma-4 Reasoning Thoughts and Content
    // Parse blocks: thoughts, tool_calls, observations and text
    function parseMessageBlocks(text) {
        if (!text) return [{ type: 'text', content: '' }];
        const blocks = [];
        let remaining = text;
        while (remaining.length > 0) {
            let markers = [
                { type: 'thought', tag: '<|channel>thought\n', endTag: '<channel|>', idx: remaining.indexOf('<|channel>thought\n') },
                { type: 'tool_call', tag: '<|tool_call>', endTag: '<tool_call|>', idx: remaining.indexOf('<|tool_call>') },
                { type: 'observation', tag: '\nObservation (', endTag: '\n', idx: remaining.indexOf('\nObservation (') }
            ].filter(m => m.idx !== -1).sort((a, b) => a.idx - b.idx);
            
            if (markers.length === 0) {
                if (remaining.trim()) blocks.push({ type: 'text', content: remaining });
                break;
            }
            const firstMarker = markers[0];
            if (firstMarker.idx > 0) {
                const preText = remaining.substring(0, firstMarker.idx);
                if (preText.trim()) blocks.push({ type: 'text', content: preText });
            }
            remaining = remaining.substring(firstMarker.idx);
            
            if (firstMarker.type === 'observation') {
                let nextThought = remaining.indexOf('<|channel>thought', 14);
                let nextTool = remaining.indexOf('<|tool_call>', 14);
                let endIdx = -1;
                if (nextThought !== -1 && nextTool !== -1) endIdx = Math.min(nextThought, nextTool);
                else if (nextThought !== -1) endIdx = nextThought;
                else if (nextTool !== -1) endIdx = nextTool;
                
                let obsText = endIdx !== -1 ? remaining.substring(0, endIdx) : remaining;
                remaining = endIdx !== -1 ? remaining.substring(endIdx) : "";
                
                const match = obsText.match(/^\nObservation \((.*?)\):\s*([\s\S]*)/);
                if (match) {
                    blocks.push({ type: 'tool_result', tool: match[1], content: match[2].trim() });
                } else {
                    blocks.push({ type: 'text', content: obsText });
                }
            } else {
                let endIdx = remaining.indexOf(firstMarker.endTag);
                if (endIdx !== -1) {
                    let content = remaining.substring(firstMarker.tag.length, endIdx).trim();
                    if (firstMarker.type === 'tool_call') {
                        let toolMatch = content.match(/^call:?([a-zA-Z0-9_\-/]+)?\s*(\{.*\})/s);
                        if (toolMatch) {
                            blocks.push({ type: 'tool_use', tool: toolMatch[1] || 'tool', args: toolMatch[2], streaming: false });
                        } else {
                            blocks.push({ type: 'tool_use', tool: 'tool', args: content, streaming: false });
                        }
                    } else {
                        blocks.push({ type: 'thought', content: content });
                    }
                    remaining = remaining.substring(endIdx + firstMarker.endTag.length);
                } else {
                    let content = remaining.substring(firstMarker.tag.length).trim();
                    if (firstMarker.type === 'tool_call') {
                        let toolMatch = content.match(/^call:?([a-zA-Z0-9_\-/]+)?\s*(\{.*\})/s);
                        if (toolMatch) {
                            blocks.push({ type: 'tool_use', tool: toolMatch[1] || 'tool', args: toolMatch[2], streaming: true });
                        } else {
                            blocks.push({ type: 'tool_use', tool: 'tool', args: content, streaming: true });
                        }
                    } else {
                        blocks.push({ type: 'thought', content: content, streaming: true });
                    }
                    remaining = "";
                }
            }
        }
        
        let grouped = [];
        for (let i = 0; i < blocks.length; i++) {
            let b = blocks[i];
            if (b.type === 'tool_use') {
                let next = blocks[i+1];
                if (next && next.type === 'tool_result' && next.tool === b.tool) {
                    grouped.push({ type: 'tool_invocation', tool: b.tool, args: b.args, result: next.content, streaming: b.streaming });
                    i++;
                } else {
                    grouped.push({ type: 'tool_invocation', tool: b.tool, args: b.args, result: null, streaming: b.streaming });
                }
            } else {
                grouped.push(b);
            }
        }
        return grouped;
    }

    function toggleBlock(key) {
        const currentVal = expandedThoughts[key] !== undefined 
            ? expandedThoughts[key] 
            : !defaultReasoningCollapsed;
        expandedThoughts[key] = !currentVal;
    }

    function toggleToolArgs(key) {
        expandedToolArgs[key] = !(expandedToolArgs[key] !== undefined ? expandedToolArgs[key] : false);
    }
    
    function toggleToolResult(key) {
        expandedToolResults[key] = !(expandedToolResults[key] !== undefined ? expandedToolResults[key] : false);
    }

    // Markdown Formatter Helper
    function formatMessage(content) {
        if (!content) return '';
        try {
            const parsed = marked.parse(content);
            return DOMPurify.sanitize(parsed);
        } catch (e) {
            console.error("Markdown parsing error", e);
            return content;
        }
    }
</script>

<div class="flex h-screen w-screen bg-[var(--bg-main)] font-sans text-[var(--text-primary)] overflow-hidden relative select-none {theme === 'light' ? 'light-theme' : ''}">
    
    <!-- LOADING MODAL OVERLAY -->
    {#if showLoader}
        <div class="absolute inset-0 bg-[var(--bg-main)]/90 backdrop-blur-md flex flex-col items-center justify-center z-50 animate-fade-in select-text">
            <div class="relative w-16 h-16 flex items-center justify-center mb-5">
                <div class="absolute border-4 border-indigo-500/20 rounded-full w-full h-full"></div>
                <div class="absolute border-4 border-t-indigo-500 border-r-indigo-500/0 border-b-indigo-500/0 border-l-indigo-500/0 rounded-full w-full h-full animate-spin"></div>
            </div>
            <h3 class="text-lg font-semibold text-white mb-2">{loaderTitle}</h3>
            <p class="text-sm text-slate-400 text-center max-w-sm px-6 leading-relaxed">{loaderSubtitle}</p>
        </div>
    {/if}

    <!-- MODEL LOADER MODAL -->
    {#if showModelLoaderModal}
        <div role="dialog" aria-modal="true" tabindex="-1" class="absolute inset-0 bg-[var(--bg-main)]/80 backdrop-blur-sm flex items-center justify-center z-40 animate-fade-in select-text outline-none" onclick={(e) => { if (e.target === e.currentTarget) showModelLoaderModal = false; }} onkeydown={(e) => { if (e.key === 'Escape') showModelLoaderModal = false; }}>
            <div class="bg-[var(--bg-panel)] border border-[var(--border-color)] rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[85vh] animate-scale-up">
                
                <!-- Modal Header -->
                <div class="flex items-center justify-between p-5 border-b border-[var(--border-color)] bg-[var(--bg-input)]/40">
                    <div>
                        <h3 class="text-md font-bold text-[var(--text-primary)]">Model Loader Settings</h3>
                        <p class="text-xs text-[var(--text-muted)] mt-0.5">Configure and initialize MLX weights on local Apple Silicon hardware</p>
                    </div>
                    <button type="button" onclick={() => showModelLoaderModal = false} aria-label="Close modal" title="Close modal" class="w-8 h-8 rounded-lg hover:bg-[var(--bg-hover)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition">
                        <span class="material-symbols-outlined text-[16px]">close</span>
                    </button>
                </div>
                
                <!-- Modal Content (Scrollable) -->
                <div class="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
                    <!-- Top section: Status & Presets -->
                    <div class="grid grid-cols-2 gap-4">
                        <!-- Left pane: Connection & Active status -->
                        <div class="flex flex-col gap-3 bg-[var(--bg-input)] border border-[var(--border-color)] rounded-xl p-4">
                            <div class="text-[9px] text-[var(--text-muted)] font-bold uppercase tracking-wider">Gateway Status</div>
                            <div class="flex items-center justify-between text-xs">
                                <div class="flex items-center gap-2">
                                    <span class="w-2.5 h-2.5 rounded-full {isConnected ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50 animate-pulse' : 'bg-rose-500'}"></span>
                                    <span class="font-semibold text-[var(--text-primary)]">{isConnected ? 'Server Connected' : 'Server Offline'}</span>
                                </div>
                                <span class="font-mono text-[var(--text-muted)] text-[10px]">127.0.0.1:{serverPort}</span>
                            </div>
                            
                            <div class="border-t border-[var(--border-color)] my-1"></div>
                            
                            <div class="text-[9px] text-[var(--text-muted)] font-bold uppercase tracking-wider">Active Loaded Model</div>
                            <div class="text-[13px] font-bold text-[var(--text-primary)] truncate">{activeModel}</div>
                            <div class="flex items-center gap-2">
                                <span class="bg-[var(--accent-color)]/15 text-[var(--accent-color)] text-[9px] px-2 py-0.5 rounded font-mono uppercase">{activeType}</span>
                                {#if isModelLoaded}
                                    <span class="bg-emerald-500/10 border border-emerald-500/20 text-emerald-500 text-[9px] px-2 py-0.5 rounded">Ready</span>
                                    <button type="button" onclick={() => { unloadModel(); showModelLoaderModal = false; }} class="ml-auto bg-transparent border border-rose-500/20 hover:bg-rose-500/10 text-rose-500 rounded-md px-2 py-1 text-[10px] transition font-medium">Unload</button>
                                {:else}
                                    <span class="bg-[var(--bg-panel)] border border-[var(--border-color)] text-[var(--text-muted)] text-[9px] px-2 py-0.5 rounded">No Model</span>
                                {/if}
                            </div>
                        </div>
                        
                        <!-- Right pane: Quick Presets selector -->
                        <div class="flex flex-col gap-3 bg-[var(--bg-input)] border border-[var(--border-color)] rounded-xl p-4">
                            <div class="text-[9px] text-[var(--text-muted)] font-bold uppercase tracking-wider">Quick Presets</div>
                            <label class="flex flex-col gap-1.5 flex-1">
                                <span class="text-[10px] text-[var(--text-secondary)] select-none">Choose a default configuration:</span>
                                <select value={presetKey} onchange={handlePresetChange} class="w-full bg-[var(--bg-panel)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs focus:border-[var(--accent-color)] focus:outline-none transition font-sans">
                                    <option value="">-- Custom Loader Setup --</option>
                                    {#each Object.entries(PRESETS) as [key, preset]}
                                        <option value={key}>{preset.name}</option>
                                    {/each}
                                </select>
                            </label>
                            <p class="text-[10px] text-[var(--text-muted)] italic">Presets auto-populate configuration paths below.</p>
                        </div>
                    </div>
                    
                    <!-- Advanced Configuration Form -->
                    <div class="flex flex-col gap-4 border-t border-[var(--border-color)] pt-5">
                        <div class="text-[11px] font-bold uppercase tracking-wider text-[var(--accent-color)]">Advanced Parameters</div>
                        
                        <div class="grid grid-cols-2 gap-4">
                            <label class="flex flex-col gap-1.5">
                                <span class="text-[10px] font-bold text-[var(--text-secondary)]">Model Architecture</span>
                                <select bind:value={modelType} onchange={() => presetKey = ''} class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs focus:border-[var(--accent-color)] focus:outline-none transition">
                                    <option value="recurrentgemma">RecurrentGemma</option>
                                    <option value="t5v2">T5v2 (Encoder-Decoder)</option>
                                    <option value="standard">Standard mlx-lm</option>
                                </select>
                            </label>
                            
                            <label class="flex flex-col gap-1.5">
                                <span class="text-[10px] font-bold text-[var(--text-secondary)]">Attention Sliding Window (Optional)</span>
                                <input type="number" bind:value={attnWindow} placeholder="Default config value" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs font-mono focus:border-[var(--accent-color)] focus:outline-none transition" />
                            </label>
                        </div>
                        
                        <label class="flex flex-col gap-1.5">
                            <span class="text-[10px] font-bold text-[var(--text-secondary)]">Config Path</span>
                            <input type="text" bind:value={configPath} oninput={() => presetKey = ''} placeholder="e.g. models/gemma-2b/config.json" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs font-mono focus:border-[var(--accent-color)] focus:outline-none transition" />
                        </label>
                        
                        <label class="flex flex-col gap-1.5">
                            <span class="text-[10px] font-bold text-[var(--text-secondary)]">Weights Path</span>
                            <input type="text" bind:value={weightsPath} oninput={() => presetKey = ''} placeholder="e.g. weights/gemma-2b/weights.safetensors" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs font-mono focus:border-[var(--accent-color)] focus:outline-none transition" />
                        </label>
                        
                        <label class="flex flex-col gap-1.5">
                            <span class="text-[10px] font-bold text-[var(--text-secondary)]">Tokenizer / HuggingFace Path</span>
                            <input type="text" bind:value={tokenizerPath} oninput={() => presetKey = ''} placeholder="e.g. storage/models/gemma-2b" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs font-mono focus:border-[var(--accent-color)] focus:outline-none transition" />
                        </label>
                        
                        <label class="flex flex-col gap-1.5">
                            <span class="text-[10px] font-bold text-[var(--text-secondary)]">Lora Adapter Path (Optional)</span>
                            <input type="text" bind:value={adapterPath} placeholder="e.g. storage/models/lora-adapters" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs font-mono focus:border-[var(--accent-color)] focus:outline-none transition" />
                        </label>
                        
                        <label class="flex flex-col gap-1.5">
                            <span class="text-[10px] font-bold text-[var(--text-secondary)]">Chat Template Path (Optional)</span>
                            <input type="text" bind:value={chatTemplatePath} oninput={() => presetKey = ''} placeholder="e.g. storage/models/gemma-4-12B-it/chat_template.jinja" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs font-mono focus:border-[var(--accent-color)] focus:outline-none transition" />
                        </label>
                        
                        <label class="flex flex-col gap-1.5">
                            <span class="text-[10px] font-bold text-[var(--text-secondary)]">Draft Speculative Model Path (Optional)</span>
                            <input type="text" bind:value={draftModelPath} oninput={() => presetKey = ''} placeholder="e.g. storage/weights/gemma-4-12B-it-assistant-mxfp4" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs font-mono focus:border-[var(--accent-color)] focus:outline-none transition" />
                        </label>
                        
                        <div class="grid grid-cols-2 gap-4">
                            <label class="flex flex-col gap-1.5">
                                <span class="text-[10px] font-bold text-[var(--text-secondary)]">Draft Kind</span>
                                <select bind:value={draftKind} onchange={() => presetKey = ''} class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs focus:border-[var(--accent-color)] focus:outline-none transition">
                                    <option value="">Auto (DFlash)</option>
                                    <option value="mtp">MTP (Unified Gemma-4)</option>
                                    <option value="dflash">DFlash (Standard Speculative)</option>
                                </select>
                            </label>
                            
                            <label class="flex flex-col gap-1.5">
                                <span class="text-[10px] font-bold text-[var(--text-secondary)]">Draft Block Size</span>
                                <input type="number" bind:value={draftBlockSize} oninput={() => presetKey = ''} placeholder="e.g. 4" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs font-mono focus:border-[var(--accent-color)] focus:outline-none transition" />
                            </label>
                        </div>
                    </div>
                </div>
                
                <!-- Modal Footer -->
                <div class="p-5 border-t border-[var(--border-color)] bg-[var(--bg-input)]/40 flex justify-end gap-3">
                    <button type="button" onclick={() => showModelLoaderModal = false} class="bg-transparent border border-[var(--border-color)] hover:border-[var(--text-secondary)] hover:bg-[var(--bg-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] rounded-lg px-4 py-2 text-xs transition">
                        Cancel
                    </button>
                    <button type="button" onclick={() => { loadModel(); showModelLoaderModal = false; }} disabled={!isConnected} class="bg-[var(--accent-color)] hover:bg-[var(--accent-hover)] disabled:bg-[var(--bg-panel)] disabled:text-[var(--text-muted)] text-white rounded-lg px-5 py-2 text-xs font-medium transition active:scale-[0.98] shadow-md shadow-[var(--accent-color)]/10">
                        Load Configured Model
                    </button>
                </div>
            </div>
        </div>
    {/if}

    <!-- LEFT SIDEBAR: CHAT HISTORY MANAGER -->
    {#if isLeftSidebarOpen}
        <aside class="w-[320px] bg-[var(--bg-sidebar)] border-r border-[var(--border-color)] flex flex-col h-full overflow-hidden shrink-0 select-text">
            
            <!-- Chat Sidebar Container -->
            <div class="flex-1 overflow-y-auto p-4 flex flex-col min-h-0">
                <div class="flex items-center justify-between mb-4 select-none">
                    <h2 class="text-xs font-bold uppercase tracking-wider text-[var(--text-secondary)]">Chats</h2>
                    <button type="button" onclick={() => createNewChat()} title="New Chat" aria-label="New Chat" class="w-7 h-7 bg-[var(--accent-color)] hover:bg-[var(--accent-hover)] text-white rounded-lg transition shadow-md shadow-black/10 active:scale-95 flex items-center justify-center">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                        </svg>
                    </button>
                </div>

                <!-- Search -->
                <div class="mb-3">
                    <input type="text" bind:value={searchQuery} placeholder="Search chats..." class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] placeholder-[var(--text-muted)] px-3 py-1.5 text-xs focus:border-[var(--accent-color)] focus:outline-none transition" />
                </div>

                <!-- New Folder Button -->
                <button type="button" onclick={() => createFolder()} class="w-full flex items-center gap-2 px-3 py-1.5 text-xs font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] rounded-lg transition border border-[var(--border-color)] mb-3">
                    <span class="material-symbols-outlined text-[14px] text-[var(--text-muted)]">dns</span>
                    <span>New Folder</span>
                </button>

                <!-- Scrollable list -->
                <div class="flex-1 overflow-y-auto pr-1 flex flex-col gap-1 min-h-0">
                    <!-- 1. Folders -->
                    {#each folders.filter(f => f.parentId === null) as folder}
                        {@render renderFolder(folder, 0)}
                    {/each}

                    <!-- 2. Uncategorized Chats -->
                    {#each sessions.filter(s => s.folderId === null && s.title.toLowerCase().includes(searchQuery.toLowerCase())).sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0)) as session}
                        {@render renderChat(session, 0)}
                    {/each}
                </div>
            </div>

            <!-- Left Sidebar Footer: Memory Indicator -->
            <div class="p-4 border-t border-[var(--border-color)] bg-[var(--bg-panel)]/30 text-xs text-[var(--text-secondary)] select-none shrink-0 flex flex-col gap-2">
                {#if isConnected}
                    <div class="flex flex-col gap-1.5">
                        <div class="flex justify-between items-center text-[10px] uppercase font-bold tracking-wider text-[var(--text-muted)]">
                            <span>GPU VRAM</span>
                            {#if gpuLimitGb > 0}
                                <span class="font-mono text-[var(--text-secondary)] font-semibold">{activeMemGb.toFixed(1)} / {gpuLimitGb.toFixed(1)} GB</span>
                            {:else}
                                <span class="font-mono text-[var(--text-secondary)] font-semibold">{activeMemGb.toFixed(1)} GB</span>
                            {/if}
                        </div>
                        <div class="w-full h-1.5 bg-[var(--border-color)] rounded-full overflow-hidden flex">
                            <div class="h-full bg-[var(--accent-color)] transition-all duration-300" style="width: {gpuLimitGb > 0 ? Math.min(100, (activeMemGb / gpuLimitGb) * 100) : 0}%"></div>
                            <div class="h-full bg-[var(--accent-color)]/35 transition-all duration-300" style="width: {gpuLimitGb > 0 ? Math.min(100 - (activeMemGb / gpuLimitGb) * 100, (cacheMemGb / gpuLimitGb) * 100) : 0}%"></div>
                        </div>
                        <div class="flex justify-between text-[9px] text-[var(--text-muted)] mt-0.5 leading-none">
                            <span>Cache: {cacheMemGb.toFixed(1)} GB</span>
                            <span>Peak: {peakMemGb.toFixed(1)} GB</span>
                        </div>
                    </div>

                    {#if systemRamUsedGb !== null && systemRamTotalGb !== null}
                        <div class="border-t border-[var(--border-color)]/50 my-1"></div>
                        <div class="flex flex-col gap-1.5">
                            <div class="flex justify-between items-center text-[10px] uppercase font-bold tracking-wider text-[var(--text-muted)]">
                                <span>System RAM</span>
                                <span class="font-mono text-[var(--text-secondary)] font-semibold">{systemRamUsedGb.toFixed(1)} / {systemRamTotalGb.toFixed(0)} GB</span>
                            </div>
                            <div class="w-full h-1.5 bg-[var(--border-color)] rounded-full overflow-hidden">
                                <div class="h-full bg-[var(--text-secondary)] rounded-full transition-all duration-300" style="width: {Math.min(100, Math.round((systemRamUsedGb / systemRamTotalGb) * 100))}%"></div>
                            </div>
                        </div>
                    {/if}
                {:else}
                    <div class="flex items-center gap-1.5 text-[10px] text-[var(--text-muted)] italic">
                        <span class="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse"></span>
                        <span>Gateway offline</span>
                    </div>
                {/if}
            </div>

        </aside>
    {/if}

    <!-- CENTER AREA: CHAT WORKSPAC    <!-- CENTER AREA: CHAT WORKSPACE -->
    <main class="flex-1 flex flex-col h-full min-w-0 bg-[var(--bg-main)] select-text relative">
        
        <!-- Header status bar -->
        <header class="h-[56px] border-b border-[var(--border-color)] flex items-center justify-between px-6 bg-[var(--bg-panel)]/80 backdrop-blur-md shrink-0 select-none">
            <!-- Left: Sidebar toggle + Active Chat name -->
            <div class="flex items-center gap-3">
                <button type="button" onclick={() => isLeftSidebarOpen = !isLeftSidebarOpen} title="Toggle Sidebar" class="p-2 border border-[var(--border-color)] hover:border-[var(--text-secondary)] hover:bg-[var(--bg-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] rounded-lg transition active:scale-[0.98]">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
                    </svg>
                </button>
                
                <button type="button" onclick={() => {
                    const session = sessions.find(s => s.id === currentChatId);
                    if (session) {
                        const newTitle = prompt("Enter new chat title:", session.title);
                        if (newTitle && newTitle.trim()) {
                            session.title = newTitle.trim();
                            saveToLocalStorage();
                        }
                    }
                }} class="text-left group/title">
                    <div class="text-[9px] uppercase font-bold tracking-wider text-[var(--text-muted)] leading-none">Active Chat</div>
                    <div class="text-[13px] font-semibold text-[var(--text-primary)] group-hover/title:text-[var(--accent-color)] transition-colors flex items-center gap-1.5 mt-0.5">
                        <span class="truncate max-w-[200px]">{sessions.find(s => s.id === currentChatId)?.title || 'New Conversation'}</span>
                        <span class="opacity-0 group-hover/title:opacity-100 transition-opacity">
                            <svg class="w-3 h-3 text-[var(--text-muted)]" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L6.832 19.82a4.5 4.5 0 01-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 011.13-1.897L16.863 4.487zm0 0L19.5 7.125" />
                            </svg>
                        </span>
                    </div>
                </button>
            </div>
            
            <!-- Center: Model Select Capsule -->
            <div class="flex items-center justify-center">
                <button type="button" onclick={() => showModelLoaderModal = true} class="flex items-center gap-2.5 px-4.5 py-2 bg-[var(--bg-panel)] border border-[var(--border-color)] hover:border-[var(--accent-color)]/50 hover:bg-[var(--bg-hover)] rounded-full text-xs font-semibold text-[var(--text-primary)] transition active:scale-[0.98] shadow-lg shadow-black/10">
                    {#if isModelLoaded}
                        <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50"></span>
                        <span class="truncate max-w-[150px]">{activeModel}</span>
                        <span class="bg-[var(--accent-color)]/15 text-[var(--accent-color)] text-[8px] px-1.5 py-0.5 rounded font-mono uppercase tracking-wider">{activeType}</span>
                    {:else}
                        <span class="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse"></span>
                        <span class="text-[var(--text-muted)] font-medium">Select model to load...</span>
                    {/if}
                    <span class="material-symbols-outlined text-[12px] text-[var(--text-muted)]">expand_more</span>
                </button>
            </div>
            
            <!-- Right: Actions -->
            <div class="flex items-center gap-2">
                <button type="button" onclick={clearChats} disabled={chatHistory.length === 0} class="bg-transparent border border-[var(--border-color)] hover:border-[var(--text-secondary)] hover:bg-[var(--bg-hover)] disabled:border-[var(--border-color)]/30 disabled:text-[var(--text-muted)] text-[var(--text-secondary)] rounded-lg px-3.5 py-2 text-xs transition active:scale-[0.98]">
                    Clear Chats
                </button>
                <button 
                    type="button" 
                    onclick={() => { theme = theme === 'dark' ? 'light' : 'dark'; saveToLocalStorage(); }} 
                    title="Toggle Theme" 
                    class="p-2 border border-[var(--border-color)] hover:border-[var(--text-secondary)] hover:bg-[var(--bg-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] rounded-lg transition active:scale-[0.98]"
                    aria-label="Toggle Theme"
                >
                    {#if theme === 'dark'}
                        <!-- Moon SVG -->
                        <span class="material-symbols-outlined text-[16px]">dark_mode</span>
                    {:else}
                        <!-- Sun SVG -->
                        <span class="material-symbols-outlined text-[16px]">light_mode</span>
                    {/if}
                </button>
                <button type="button" onclick={() => isRightSidebarOpen = !isRightSidebarOpen} title="Toggle Parameters" class="p-2 border border-[var(--border-color)] hover:border-[var(--text-secondary)] hover:bg-[var(--bg-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] rounded-lg transition active:scale-[0.98]">
                    <span class="material-symbols-outlined text-[16px]">tune</span>
                </button>
            </div>
        </header>

        <!-- Messages scroll viewport -->
        <div bind:this={messagesContainer} class="flex-1 min-height-0 overflow-y-auto px-8 py-6 flex flex-col gap-6 {isGenerating ? 'generating' : ''}">
            {#if chatHistory.length === 0}
                <div class="flex-1 flex flex-col items-center justify-center text-center p-8 select-none">
                    <div class="w-16 h-16 rounded-2xl bg-[var(--bg-panel)] border border-[var(--border-color)] flex items-center justify-center text-2xl mb-4 text-[var(--text-secondary)]">
                        <span class="material-symbols-outlined text-[32px] text-[var(--text-muted)]">dashboard_customize</span>
                    </div>
                    <h3 class="text-sm font-semibold text-[var(--text-primary)] mb-1">Unified MLX Dashboard</h3>
                    <p class="text-xs text-[var(--text-muted)] max-w-sm leading-relaxed">Select a preset on the top header and load a model to begin testing. Switch between RecurrentGemma, T5, and transformer models instantly.</p>
                </div>
            {:else}
                {#each chatHistory as msg, index}
                    <div class="group relative flex flex-col w-full border-b border-[var(--border-color)]/30 pb-6 animate-fade-in">
                        <!-- Message Header -->
                        <div class="flex items-center justify-between mb-2 select-none">
                            <div class="flex items-center gap-2">
                                {#if msg.role === 'user'}
                                    <div class="w-6 h-6 rounded-md bg-[var(--bg-panel)] border border-[var(--border-color)] flex items-center justify-center text-[var(--text-secondary)]">
                                        <span class="material-symbols-outlined text-[14px]">person</span>
                                    </div>
                                    <span class="text-xs font-semibold text-[var(--text-primary)]">User</span>
                                {:else}
                                    <div class="w-6 h-6 rounded-md bg-[var(--accent-color)]/10 border border-[var(--accent-color)]/20 flex items-center justify-center text-[var(--accent-color)]">
                                        <span class="material-symbols-outlined text-[14px]">smart_toy</span>
                                    </div>
                                    <span class="text-xs font-semibold text-[var(--text-primary)]">
                                        {isModelLoaded ? activeModel : 'Assistant'}
                                    </span>
                                    {#if activeType}
                                        <span class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-[var(--bg-panel)] border border-[var(--border-color)] text-[var(--text-muted)] font-semibold uppercase tracking-wider select-none">
                                            {activeType}
                                        </span>
                                    {/if}
                                {/if}
                            </div>
                            
                            <!-- Action buttons moved to bottom -->
                        </div>

                        <!-- Message Body -->
                        <div class="pl-8 pr-4">
                            {#if editingIndex === index}
                                <div class="flex flex-col w-full gap-2 mt-2">
                                    <textarea use:autoresize bind:value={editBuffer} onkeydown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveEdit(index); } }} class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-xl text-[var(--text-primary)] px-4 py-3 text-[14px] leading-relaxed focus:border-[var(--accent-color)] focus:outline-none transition resize-none h-auto min-h-[80px] shadow-sm"></textarea>
                                    <div class="flex gap-2 justify-end select-none mt-1">
                                        <button onclick={cancelEdit} class="bg-transparent border border-[var(--border-color)] hover:border-[var(--text-secondary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] rounded-lg px-4 py-1.5 text-xs transition">Cancel</button>
                                        <button onclick={() => saveEdit(index)} class="bg-[var(--accent-color)] hover:bg-[var(--accent-hover)] text-white font-medium rounded-lg px-4 py-1.5 text-xs transition">{msg.role === 'assistant' ? 'Save' : 'Save & Submit'}</button>
                                    </div>
                                </div>
                            {:else}
                                <div class="text-[14px] text-[var(--text-primary)] leading-relaxed select-text {msg.role === 'user' ? 'whitespace-pre-wrap' : ''}">
                                    {#if msg.role === 'user'}
                                        {msg.content}
                                    {:else}
                                        {#if msg.isPlaceholder}
                                            <div class="flex items-center gap-1.5 py-3 select-none">
                                                <span class="w-1.5 h-1.5 bg-[var(--text-muted)] rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                                                <span class="w-1.5 h-1.5 bg-[var(--text-muted)] rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                                                <span class="w-1.5 h-1.5 bg-[var(--text-muted)] rounded-full animate-bounce"></span>
                                            </div>
                                        {:else}
                                            {@const blocks = parseMessageBlocks(msg.content)}
                                            {#each blocks as block, blockIndex}
                                                {#if block.type === 'thought'}
                                                    {#if !hideReasoningBlocks}
                                                        {@const key = index + '-' + blockIndex}
                                                        {@const isExpanded = expandedThoughts[key] !== undefined ? expandedThoughts[key] : !defaultReasoningCollapsed}
                                                        <div class="mb-4 rounded-xl border border-[var(--border-color)] bg-[var(--bg-panel)]/40 overflow-hidden text-xs max-w-4xl">
                                                            <!-- Header / Toggle Bar -->
                                                            <button 
                                                                type="button" 
                                                                onclick={() => toggleBlock(key)} 
                                                                class="w-full flex items-center justify-between px-3.5 py-2.5 text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition select-none font-medium text-[11px]"
                                                            >
                                                                <div class="flex items-center gap-2">
                                                                    <span class="material-symbols-outlined text-[14px] text-[var(--text-muted)]">psychology</span>
                                                                    <span>Thinking Process</span>
                                                                    {#if block.streaming && isGenerating && index === chatHistory.length - 1}
                                                                        <span class="w-1.5 h-1.5 rounded-full bg-[var(--accent-color)] animate-pulse"></span>
                                                                    {/if}
                                                                </div>
                                                                <div class="flex items-center gap-1.5 text-[10.5px] text-[var(--text-muted)] font-mono">
                                                                    {isExpanded ? 'Collapse' : 'Expand'}
                                                                    <span class="material-symbols-outlined text-[12px] transition-transform duration-200 {isExpanded ? 'rotate-180' : ''}">expand_more</span>
                                                                </div>
                                                            </button>
                                                            
                                                            <!-- Thought Content -->
                                                            {#if isExpanded}
                                                                <div class="px-4 pb-3.5 pt-1.5 border-t border-[var(--border-color)]/30 text-[var(--text-secondary)] font-mono leading-relaxed max-h-[300px] overflow-y-auto whitespace-pre-wrap select-text bg-[var(--bg-input)]/20">
                                                                    {block.content}
                                                                </div>
                                                            {/if}
                                                        </div>
                                                    {/if}
                                                {:else if block.type === 'tool_invocation'}
                                                    {@const key = index + '-' + blockIndex}
                                                    {@const isExpanded = expandedThoughts[key] !== undefined ? expandedThoughts[key] : true}
                                                    <div class="mb-4 ml-4 pl-3 border-l-2 border-blue-500/30 text-[13px] max-w-4xl font-mono text-[var(--text-secondary)]">
                                                        <div class="flex items-center justify-between mb-2 select-none text-[var(--text-primary)]">
                                                            <div class="flex items-center gap-2 text-blue-500">
                                                                <span class="material-symbols-outlined text-[16px]">build</span>
                                                                <span class="font-bold tracking-wide">{block.tool}</span>
                                                                {#if block.streaming && isGenerating && index === chatHistory.length - 1}
                                                                    <span class="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse ml-1"></span>
                                                                {/if}
                                                            </div>
                                                            <button onclick={() => toggleBlock(key)} class="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[var(--bg-input)] border border-[var(--border-color)] text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition shadow-sm">
                                                                mcp/{block.tool}
                                                                <span class="material-symbols-outlined text-[12px] transition-transform duration-200 {isExpanded ? 'rotate-180' : ''}">expand_more</span>
                                                            </button>
                                                        </div>
                                                        {#if isExpanded}
                                                            <div class="flex flex-col gap-1.5">
                                                                <!-- Arguments Accordion -->
                                                                <div class="flex flex-col">
                                                                    <button onclick={() => toggleToolArgs(key)} class="flex items-start gap-1.5 text-left text-[var(--text-primary)] hover:text-[var(--accent-color)] transition group w-full">
                                                                        <span class="material-symbols-outlined text-[12px] mt-1 text-[var(--text-muted)] transition-transform {expandedToolArgs[key] ? 'rotate-90' : ''}">chevron_right</span>
                                                                        <div class="flex-1 {expandedToolArgs[key] ? '' : 'flex items-center'} overflow-hidden min-w-0">
                                                                            <span class="font-semibold whitespace-nowrap">Arguments:</span>
                                                                            <span class="ml-1.5 text-[var(--text-muted)] {expandedToolArgs[key] ? 'whitespace-pre-wrap block mt-1 bg-[var(--bg-input)]/30 p-2 rounded-lg text-[12px]' : 'truncate inline-block align-bottom'}" title={!expandedToolArgs[key] ? block.args : ''}>{block.args}</span>
                                                                        </div>
                                                                    </button>
                                                                </div>
                                                                
                                                                <!-- Results Accordion -->
                                                                {#if block.result}
                                                                    <div class="flex flex-col">
                                                                        <button onclick={() => toggleToolResult(key)} class="flex items-start gap-1.5 text-left text-[var(--text-primary)] hover:text-[var(--accent-color)] transition group w-full">
                                                                            <span class="material-symbols-outlined text-[12px] mt-1 text-[var(--text-muted)] transition-transform {expandedToolResults[key] ? 'rotate-90' : ''}">chevron_right</span>
                                                                            <div class="flex-1 {expandedToolResults[key] ? '' : 'flex items-center'} overflow-hidden min-w-0">
                                                                                <span class="font-semibold whitespace-nowrap">Result:</span>
                                                                                <span class="ml-1.5 text-[var(--text-muted)] {expandedToolResults[key] ? 'whitespace-pre-wrap block mt-1 max-h-[300px] overflow-y-auto bg-[var(--bg-input)]/30 p-2 rounded-lg text-[12px]' : 'truncate inline-block align-bottom'}" title={!expandedToolResults[key] ? block.result : ''}>{block.result}</span>
                                                                            </div>
                                                                        </button>
                                                                    </div>
                                                                {/if}
                                                            </div>
                                                        {/if}
                                                    </div>
                                                {:else if block.type === 'text' && block.content}
                                                    <div class="prose prose-sm dark:prose-invert max-w-none prose-pre:p-0 prose-pre:bg-transparent prose-pre:m-0 leading-relaxed">
                                                        {@html formatMessage(block.content)}
                                                    </div>
                                                {/if}
                                            {/each}
                                            
                                            {#if isGenerating && index === chatHistory.length - 1}
                                                {@const blocks = parseMessageBlocks(msg.content)}
                                                {#if blocks.length === 0 || (!blocks[blocks.length - 1].streaming && blocks[blocks.length - 1].type !== 'text' && blocks[blocks.length - 1].type !== 'thought' && blocks[blocks.length - 1].type !== 'tool_invocation')}
                                                    <div class="flex items-center gap-2 text-[12px] text-[var(--text-muted)] italic py-1 select-none animate-pulse">
                                                        <span>Formulating response...</span>
                                                    </div>
                                                {/if}
                                            {/if}

                                            <!-- Render tool approval card if pending -->
                                            {#if msg.pendingApproval}
                                                <div class="mt-4 p-4 rounded-xl border border-amber-500/20 bg-amber-500/5 max-w-xl select-none animate-fade-in">
                                                    <div class="flex items-center gap-2 text-xs font-semibold text-amber-600 dark:text-amber-400 mb-2">
                                                        <span>🛡️ Tool Call Authorization Required</span>
                                                    </div>
                                                    <div class="text-xs text-[var(--text-primary)] font-mono bg-[var(--bg-input)]/50 p-2.5 rounded-lg border border-[var(--border-color)]/30 mb-3 overflow-x-auto select-text">
                                                        <span class="font-bold text-[var(--text-secondary)]">{msg.pendingApproval.tool_name}</span>({JSON.stringify(msg.pendingApproval.arguments)})
                                                    </div>
                                                    <div class="flex gap-2">
                                                        <button 
                                                            type="button" 
                                                            onclick={() => handleApprovalDecision(index, msg.pendingApproval.approval_id, 'allow')} 
                                                            class="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 active:scale-[0.98] text-white text-[11px] font-semibold rounded-lg shadow-md transition"
                                                        >
                                                            Allow Execution
                                                        </button>
                                                        <button 
                                                            type="button" 
                                                            onclick={() => handleApprovalDecision(index, msg.pendingApproval.approval_id, 'reject')} 
                                                            class="px-3 py-1.5 bg-rose-600 hover:bg-rose-500 active:scale-[0.98] text-white text-[11px] font-semibold rounded-lg shadow-md transition"
                                                        >
                                                            Reject
                                                        </button>
                                                    </div>
                                                </div>
                                            {/if}

                                            <!-- Render decision status -->
                                            {#if msg.approvedStatus === 'allow'}
                                                <div class="mt-3 flex items-center gap-1.5 text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold select-none">
                                                    <span>✓ Tool call executed</span>
                                                </div>
                                            {:else if msg.approvedStatus === 'reject'}
                                                <div class="mt-3 flex items-center gap-1.5 text-[11px] text-rose-600 dark:text-rose-400 font-semibold select-none">
                                                    <span>✕ Tool call rejected by user</span>
                                                </div>
                                            {/if}
                                        {/if}
                                    {/if}
                                </div>
                                
                                <!-- Stats & Action Buttons -->
                                <div class="mt-4 flex flex-col gap-2 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity duration-150 {isGenerating ? '!hidden' : ''}">
                                    {#if msg.role === 'assistant' && msg.stats && msg.stats.tokens > 0}
                                        <div class="flex items-center gap-2 select-none mb-1">
                                            <div class="text-[var(--text-muted)] flex items-center justify-center p-1" title="Generation Stats">
                                                <span class="material-symbols-outlined text-[14px]">lightbulb</span>
                                            </div>
                                            <div class="flex items-center gap-1.5 flex-wrap">
                                                <span class="bg-[var(--bg-panel)] border border-[var(--border-color)] text-[var(--text-secondary)] text-[10px] font-medium px-2.5 py-1 rounded-full flex items-center gap-1.5">
                                                    <span class="material-symbols-outlined text-[12px]">bolt</span>
                                                    {msg.stats.tokPerSec} tok/sec
                                                </span>
                                                <span class="bg-[var(--bg-panel)] border border-[var(--border-color)] text-[var(--text-secondary)] text-[10px] font-medium px-2.5 py-1 rounded-full flex items-center gap-1.5">
                                                    <span class="material-symbols-outlined text-[12px]">database</span>
                                                    {msg.stats.tokens} tokens
                                                </span>
                                                <span class="bg-[var(--bg-panel)] border border-[var(--border-color)] text-[var(--text-secondary)] text-[10px] font-medium px-2.5 py-1 rounded-full flex items-center gap-1.5" title="Time to First Token (TTFT)">
                                                    <span class="material-symbols-outlined text-[12px]">timer</span>
                                                    {msg.stats.ttft}
                                                </span>
                                                <span class="bg-[var(--bg-panel)] border border-[var(--border-color)] text-[var(--text-secondary)] text-[10px] font-medium px-2.5 py-1 rounded-full">Stop reason: {msg.stats.stopReason}</span>
                                            </div>
                                        </div>
                                    {/if}

                                    <div class="flex items-center gap-1.5">
                                        {#if msg.alternatives && msg.alternatives.length > 1}
                                            <div class="flex items-center gap-1 px-1.5 text-xs text-[var(--text-muted)] border border-[var(--border-color)] rounded-md mr-1 select-none">
                                                <button onclick={() => switchAlternative(index, -1)} disabled={(msg.altIndex || 0) === 0} class="p-0.5 hover:text-[var(--text-primary)] disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center">
                                                    <span class="material-symbols-outlined text-[14px]">chevron_left</span>
                                                </button>
                                                <span class="font-mono text-[10px]">{(msg.altIndex || 0) + 1}/{msg.alternatives.length}</span>
                                                <button onclick={() => switchAlternative(index, 1)} disabled={(msg.altIndex || 0) === msg.alternatives.length - 1} class="p-0.5 hover:text-[var(--text-primary)] disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center">
                                                    <span class="material-symbols-outlined text-[14px]">chevron_right</span>
                                                </button>
                                            </div>
                                        {/if}

                                        {#if msg.role === 'assistant' && !msg.isPlaceholder && !msg.isError}
                                            <button onclick={() => retryMessage(index)} title="Regenerate Response" class="p-1 flex items-center justify-center hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] border border-transparent hover:border-[var(--border-color)] rounded-md transition">
                                                <span class="material-symbols-outlined text-[16px]">refresh</span>
                                            </button>
                                            {#if index === chatHistory.length - 1 && !msg.content.includes('<|tool_call>')}
                                                <button onclick={() => continueMessage(index)} title="Continue Generation" class="p-1 flex items-center justify-center hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] border border-transparent hover:border-[var(--border-color)] rounded-md transition">
                                                    <span class="material-symbols-outlined text-[16px]">play_arrow</span>
                                                </button>
                                            {/if}
                                        {/if}

                                        <button onclick={() => branchChat(index)} title="Branch Chat" class="p-1 flex items-center justify-center hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] border border-transparent hover:border-[var(--border-color)] rounded-md transition">
                                            <span class="material-symbols-outlined text-[16px]">fork_right</span>
                                        </button>

                                        {#if (!msg.isPlaceholder && !msg.isError) || msg.role === 'user'}
                                            <button onclick={() => { navigator.clipboard.writeText(msg.content); }} title="Copy to Clipboard" class="p-1 flex items-center justify-center hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] border border-transparent hover:border-[var(--border-color)] rounded-md transition">
                                                <span class="material-symbols-outlined text-[16px]">content_copy</span>
                                            </button>
                                        {/if}

                                        {#if msg.role === 'user' || (!msg.isPlaceholder && !msg.isError && !msg.content.includes('<|tool_call>'))}
                                            {#if editingIndex !== index}
                                                <button onclick={() => startEditMessage(index)} title="Edit Message" class="p-1 flex items-center justify-center hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] border border-transparent hover:border-[var(--border-color)] rounded-md transition">
                                                    <span class="material-symbols-outlined text-[16px]">edit</span>
                                                </button>
                                            {/if}
                                        {/if}

                                        <button onclick={() => deleteMessage(index)} title="Delete Message" class="p-1 flex items-center justify-center hover:bg-rose-500/10 text-[var(--text-muted)] hover:text-rose-500 border border-transparent hover:border-rose-500/30 rounded-md transition">
                                            <span class="material-symbols-outlined text-[16px]">delete</span>
                                        </button>
                                    </div>
                                </div>
                            {/if}
                        </div>
                    </div>
                {/each}
            {/if}
        </div>

        <!-- Chat Input Bar -->
        <div class="p-6 bg-[var(--bg-main)] shrink-0 flex flex-col gap-2">
            <div class="relative bg-[var(--bg-input)] border border-[var(--border-color)] rounded-2xl p-4 flex flex-col gap-3 focus-within:border-[var(--text-secondary)] transition shadow-sm min-h-[96px]">
                
                <!-- Row 1: Input -->
                <textarea use:autoresize aria-label="Prompt text" id="chat-textarea" bind:value={chatInput} onkeydown={handleKeyDown} disabled={!isModelLoaded} placeholder={isModelLoaded ? "Send a message to the model..." : "Load a model to start writing"} rows="1" class="w-full bg-transparent border-none outline-none text-[var(--text-primary)] placeholder-[var(--text-muted)] text-[15px] resize-none py-1 focus:ring-0 focus:outline-none h-auto min-h-[24px] max-h-[40vh] leading-relaxed"></textarea>
                
                <!-- Row 2: Tools, Context, Send -->
                <div class="flex items-center justify-between mt-auto">
                    <!-- Left Tools -->
                    <div class="flex items-center gap-1.5 text-[var(--text-muted)] flex-wrap">
                        <button class="p-1 hover:text-[var(--text-primary)] transition flex items-center justify-center {!isModelLoaded ? 'opacity-50 cursor-not-allowed' : ''}" title="Attach Image (Requires Vision Model)" disabled={!isModelLoaded}>
                            <span class="material-symbols-outlined text-[20px]">image</span>
                        </button>
                        
                        {#each mcpServers.filter(s => s.enabled) as server}
                            <div class="flex items-center gap-1 bg-blue-600/10 text-blue-600 dark:bg-blue-500/15 dark:text-blue-400 px-2.5 py-1.5 rounded-lg text-[12px] font-medium select-none ml-1 border border-blue-600/10 dark:border-blue-400/10">
                                {server.name.replace(/^mcp\//, '')}
                                <button class="hover:text-blue-800 dark:hover:text-blue-200 transition flex items-center justify-center ml-0.5" onclick={() => toggleMcpServer(server.name, false)} title="Disable {server.name}">
                                    <span class="material-symbols-outlined text-[12px]">close</span>
                                </button>
                            </div>
                        {/each}
                    </div>
                    
                    <!-- Right Tools -->
                    <div class="flex items-center gap-3">
                        <div class="text-[11px] font-mono text-[var(--text-muted)] select-none tracking-tight">
                            {Math.ceil((chatHistory.reduce((acc, msg) => acc + (msg.content?.length || 0), 0) + systemInstructions.length) / 4)}/8192
                        </div>
                        
                        <div class="w-[1px] h-5 bg-[var(--border-color)] opacity-70"></div>
                        
                        {#if isGenerating}
                            <button aria-label="Stop generation" onclick={stopGeneration} class="w-8 h-8 rounded-full bg-[var(--bg-panel)] hover:bg-[var(--border-color)] text-[var(--text-primary)] flex items-center justify-center transition shrink-0 active:scale-95 border border-[var(--border-color)]">
                                <span class="material-symbols-outlined text-[18px]">stop</span>
                            </button>
                        {:else}
                            <button aria-label="Send message" onclick={sendMessage} disabled={!isModelLoaded || !chatInput.trim()} class="w-8 h-8 rounded-full bg-[var(--bg-panel)] hover:bg-[var(--border-color)] disabled:opacity-40 disabled:hover:bg-[var(--bg-panel)] text-[var(--text-primary)] flex items-center justify-center transition shrink-0 active:scale-95 border border-[var(--border-color)]">
                                <span class="material-symbols-outlined text-[18px]">arrow_upward</span>
                            </button>
                        {/if}
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- RIGHT SIDEBAR: PARAMETERS & MCP -->
    {#if isRightSidebarOpen}
        <aside class="w-[320px] bg-[var(--bg-sidebar)] border-l border-[var(--border-color)] flex flex-col h-full overflow-hidden shrink-0 select-text">
            <!-- Pane Selector Header -->
            <div class="p-4 border-b border-[var(--border-color)] shrink-0 select-none">
                <div class="flex bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg p-0.5">
                    <button 
                        type="button" 
                        onclick={() => rightSidebarPane = 'params'} 
                        class="flex-1 flex items-center justify-center gap-1.5 py-1.5 text-xs font-semibold rounded-md transition-all {rightSidebarPane === 'params' ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] shadow-sm' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'}"
                        aria-label="Model Parameters"
                    >
                        <span class="material-symbols-outlined text-[14px]">format_list_bulleted</span>
                        <span>Parameters</span>
                    </button>
                    <button 
                        type="button" 
                        onclick={() => rightSidebarPane = 'mcp'} 
                        class="flex-1 flex items-center justify-center gap-1.5 py-1.5 text-xs font-semibold rounded-md transition-all {rightSidebarPane === 'mcp' ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] shadow-sm' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'}"
                        aria-label="MCP Servers"
                    >
                        <span class="material-symbols-outlined text-[14px]">memory</span>
                        <span>MCP Servers</span>
                    </button>
                </div>
            </div>

            <!-- Sidebar Scrollable Content -->
            <div class="flex-1 overflow-y-auto p-4 min-h-0 select-text">
                {#if rightSidebarPane === 'params'}
                    <div class="flex flex-col gap-4">
                        <!-- Chat Presets Loader -->
                        <div class="border border-[var(--border-color)] rounded-xl p-3 bg-[var(--bg-panel)] flex flex-col gap-2 shadow-sm">
                            <label class="flex flex-col gap-1.5">
                                <span class="text-[10px] font-bold uppercase tracking-wider text-[var(--accent-color)]">Chat Presets</span>
                                <div class="flex gap-2">
                                    <select value={selectedChatPreset} onchange={handleChatPresetChange} class="flex-1 min-w-0 bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-2 py-1.5 text-xs focus:border-[var(--accent-color)] focus:outline-none transition font-sans">
                                        <option value="">-- Manual Configuration --</option>
                                        {#each CHAT_PRESETS as preset}
                                            <option value={preset.preset_name}>{preset.preset_name}</option>
                                        {/each}
                                    </select>
                                    <div class="flex">
                                        <button 
                                            type="button" 
                                            onclick={loadChatPresets} 
                                            class="shrink-0 px-2 py-1.5 bg-[var(--bg-input)] border border-[var(--border-color)] border-r-0 rounded-l-lg text-[var(--text-primary)] text-xs hover:border-[var(--accent-color)] hover:z-10 transition flex items-center justify-center relative"
                                            title="Reload presets"
                                        >
                                            <span class="material-symbols-outlined text-[14px]">refresh</span>
                                        </button>
                                        <button 
                                            type="button" 
                                            onclick={saveChatPreset} 
                                            class="shrink-0 px-2 py-1.5 bg-[var(--bg-input)] border border-[var(--border-color)] border-r-0 rounded-none text-[var(--text-primary)] text-xs hover:border-[var(--accent-color)] hover:z-10 transition flex items-center justify-center relative"
                                            title="Save / Overwrite selected preset"
                                        >
                                            <span class="material-symbols-outlined text-[14px]">save</span>
                                        </button>
                                        <button 
                                            type="button" 
                                            onclick={saveAsNewPreset} 
                                            class="shrink-0 px-2 py-1.5 bg-[var(--bg-input)] border border-[var(--border-color)] rounded-r-lg text-[var(--text-primary)] text-xs hover:border-[var(--accent-color)] hover:z-10 transition flex items-center justify-center relative"
                                            title="Create new preset"
                                        >
                                            <span class="material-symbols-outlined text-[14px]">add_box</span>
                                        </button>
                                    </div>
                                </div>
                            </label>
                        </div>

                        <!-- Accordion: System Instructions -->
                        <div class="border border-[var(--border-color)] rounded-xl overflow-hidden bg-[var(--bg-panel)]/30">
                            <button 
                                type="button" 
                                onclick={() => collapsedSections.system = !collapsedSections.system} 
                                class="w-full flex items-center justify-between px-3.5 py-2.5 bg-[var(--bg-panel)] text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-secondary)] border-b border-[var(--border-color)] hover:bg-[var(--bg-hover)] transition"
                            >
                                <span>System Prompt</span>
                                <span class="text-[9px] text-[var(--text-muted)] transition-transform duration-200 inline-block {collapsedSections.system ? '-rotate-90' : ''}">▼</span>
                            </button>
                            {#if !collapsedSections.system}
                                <div class="p-3.5 flex flex-col gap-2">
                                    <textarea bind:value={systemInstructions} placeholder="e.g. 'You are a poetic assistant.'" class="w-full h-[80px] bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] placeholder-[var(--text-muted)] px-3 py-2 text-xs resize-none focus:border-[var(--accent-color)] focus:outline-none transition"></textarea>
                                </div>
                            {/if}
                        </div>

                        <!-- Accordion: Sampling Parameters -->
                        <div class="border border-[var(--border-color)] rounded-xl overflow-hidden bg-[var(--bg-panel)]/30">
                            <button 
                                type="button" 
                                onclick={() => collapsedSections.sampling = !collapsedSections.sampling} 
                                class="w-full flex items-center justify-between px-3.5 py-2.5 bg-[var(--bg-panel)] text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-secondary)] border-b border-[var(--border-color)] hover:bg-[var(--bg-hover)] transition"
                            >
                                <span>Sampling Params</span>
                                <span class="text-[9px] text-[var(--text-muted)] transition-transform duration-200 inline-block {collapsedSections.sampling ? '-rotate-90' : ''}">▼</span>
                            </button>
                            {#if !collapsedSections.sampling}
                                <div class="p-3.5 flex flex-col gap-4">
                                    <!-- Temperature slider -->
                                    <div class="flex flex-col gap-2">
                                        <div class="flex items-center justify-between text-xs select-none">
                                            <span class="text-[var(--text-secondary)] font-medium">Temperature</span>
                                            <input type="number" bind:value={temperature} min="0" max="2.0" step="0.01" class="w-16 bg-[var(--bg-input)] border border-[var(--border-color)] rounded px-1.5 py-0.5 text-right text-[var(--text-primary)] font-mono font-semibold text-[11px] focus:border-[var(--accent-color)] focus:outline-none transition [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none" />
                                        </div>
                                        <input type="range" bind:value={temperature} min="0" max="2.0" step="0.01" class="w-full h-1 bg-[var(--border-color)] rounded-lg appearance-none cursor-pointer accent-[var(--accent-color)]" />
                                    </div>

                                    <!-- Max tokens slider -->
                                    <div class="flex flex-col gap-2">
                                        <div class="flex items-center justify-between text-xs select-none">
                                            <span class="text-[var(--text-secondary)] font-medium">Max Tokens</span>
                                            <input type="number" bind:value={maxTokens} min="1" max="16384" step="1" class="w-20 bg-[var(--bg-input)] border border-[var(--border-color)] rounded px-1.5 py-0.5 text-right text-[var(--text-primary)] font-mono font-semibold text-[11px] focus:border-[var(--accent-color)] focus:outline-none transition [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none" />
                                        </div>
                                        <input type="range" bind:value={maxTokens} min="16" max="8192" step="16" class="w-full h-1 bg-[var(--border-color)] rounded-lg appearance-none cursor-pointer accent-[var(--accent-color)]" />
                                    </div>

                                    <!-- Top P slider -->
                                    <div class="flex flex-col gap-2">
                                        <div class="flex items-center justify-between text-xs select-none">
                                            <span class="text-[var(--text-secondary)] font-medium">Top P</span>
                                            <input type="number" bind:value={topP} min="0" max="1.0" step="0.01" class="w-16 bg-[var(--bg-input)] border border-[var(--border-color)] rounded px-1.5 py-0.5 text-right text-[var(--text-primary)] font-mono font-semibold text-[11px] focus:border-[var(--accent-color)] focus:outline-none transition [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none" />
                                        </div>
                                        <input type="range" bind:value={topP} min="0" max="1.0" step="0.01" class="w-full h-1 bg-[var(--border-color)] rounded-lg appearance-none cursor-pointer accent-[var(--accent-color)]" />
                                    </div>

                                    <!-- Top K slider -->
                                    <div class="flex flex-col gap-2">
                                        <div class="flex items-center justify-between text-xs select-none">
                                            <span class="text-[var(--text-secondary)] font-medium">Top K</span>
                                            <input type="number" bind:value={topK} min="0" max="500" step="1" class="w-16 bg-[var(--bg-input)] border border-[var(--border-color)] rounded px-1.5 py-0.5 text-right text-[var(--text-primary)] font-mono font-semibold text-[11px] focus:border-[var(--accent-color)] focus:outline-none transition [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none" />
                                        </div>
                                        <input type="range" bind:value={topK} min="0" max="100" step="1" class="w-full h-1 bg-[var(--border-color)] rounded-lg appearance-none cursor-pointer accent-[var(--accent-color)]" />
                                    </div>

                                    <!-- Repetition penalty slider -->
                                    <div class="flex flex-col gap-2">
                                        <div class="flex items-center justify-between text-xs select-none">
                                            <span class="text-[var(--text-secondary)] font-medium">Repetition Penalty</span>
                                            <input type="number" bind:value={repeatPenalty} min="0.5" max="3.0" step="0.01" class="w-16 bg-[var(--bg-input)] border border-[var(--border-color)] rounded px-1.5 py-0.5 text-right text-[var(--text-primary)] font-mono font-semibold text-[11px] focus:border-[var(--accent-color)] focus:outline-none transition [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none" />
                                        </div>
                                        <input type="range" bind:value={repeatPenalty} min="1.0" max="2.0" step="0.01" class="w-full h-1 bg-[var(--border-color)] rounded-lg appearance-none cursor-pointer accent-[var(--accent-color)]" />
                                    </div>
                                </div>
                            {/if}
                        </div>

                        <!-- Accordion: Reasoning Settings -->
                        <div class="border border-[var(--border-color)] rounded-xl overflow-hidden bg-[var(--bg-panel)]/30">
                            <button 
                                type="button" 
                                onclick={() => collapsedSections.reasoning = !collapsedSections.reasoning} 
                                class="w-full flex items-center justify-between px-3.5 py-2.5 bg-[var(--bg-panel)] text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-secondary)] border-b border-[var(--border-color)] hover:bg-[var(--bg-hover)] transition"
                            >
                                <span>Reasoning Settings</span>
                                <span class="text-[9px] text-[var(--text-muted)] transition-transform duration-200 inline-block {collapsedSections.reasoning ? '-rotate-90' : ''}">▼</span>
                            </button>
                            {#if !collapsedSections.reasoning}
                                <div class="p-3.5 flex flex-col gap-3.5">
                                    <!-- Hide reasoning blocks toggle -->
                                    <div class="flex items-center justify-between text-xs py-0.5 select-none">
                                        <span class="text-[var(--text-secondary)] font-medium">Hide Reasoning Blocks</span>
                                        <button 
                                            type="button" 
                                            onclick={() => { hideReasoningBlocks = !hideReasoningBlocks; saveToLocalStorage(); }} 
                                            class="relative inline-flex h-4.5 w-8.5 shrink-0 cursor-pointer rounded-full border border-slate-300/80 dark:border-slate-700 transition-colors duration-200 ease-in-out focus:outline-none {hideReasoningBlocks ? 'bg-[var(--accent-color)] border-transparent' : 'bg-slate-300 dark:bg-slate-800'}"
                                            role="switch"
                                            aria-checked={hideReasoningBlocks}
                                            aria-label="Hide reasoning blocks"
                                        >
                                            <span class="pointer-events-none inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out mt-[1px] ml-[1px] {hideReasoningBlocks ? 'translate-x-3.5' : 'translate-x-0'}"></span>
                                        </button>
                                    </div>

                                    <!-- Collapse reasoning by default toggle -->
                                    <div class="flex items-center justify-between text-xs py-0.5 select-none">
                                        <span class="text-[var(--text-secondary)] font-medium">Collapse by Default</span>
                                        <button 
                                            type="button" 
                                            onclick={() => { defaultReasoningCollapsed = !defaultReasoningCollapsed; saveToLocalStorage(); }} 
                                            class="relative inline-flex h-4.5 w-8.5 shrink-0 cursor-pointer rounded-full border border-slate-300/80 dark:border-slate-700 transition-colors duration-200 ease-in-out focus:outline-none {defaultReasoningCollapsed ? 'bg-[var(--accent-color)] border-transparent' : 'bg-slate-300 dark:bg-slate-800'}"
                                            role="switch"
                                            aria-checked={defaultReasoningCollapsed}
                                            aria-label="Collapse reasoning process by default"
                                        >
                                            <span class="pointer-events-none inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out mt-[1px] ml-[1px] {defaultReasoningCollapsed ? 'translate-x-3.5' : 'translate-x-0'}"></span>
                                        </button>
                                    </div>

                                    <!-- Enable reasoning in model generation -->
                                    <div class="flex items-center justify-between text-xs py-0.5 select-none">
                                        <span class="text-[var(--text-secondary)] font-medium">Enable Reasoning (Model)</span>
                                        <button 
                                            type="button" 
                                            onclick={() => { enableReasoning = !enableReasoning; saveToLocalStorage(); }} 
                                            class="relative inline-flex h-4.5 w-8.5 shrink-0 cursor-pointer rounded-full border border-slate-300/80 dark:border-slate-700 transition-colors duration-200 ease-in-out focus:outline-none {enableReasoning ? 'bg-[var(--accent-color)] border-transparent' : 'bg-slate-300 dark:bg-slate-800'}"
                                            role="switch"
                                            aria-checked={enableReasoning}
                                            aria-label="Enable reasoning in model generation"
                                        >
                                            <span class="pointer-events-none inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out mt-[1px] ml-[1px] {enableReasoning ? 'translate-x-3.5' : 'translate-x-0'}"></span>
                                        </button>
                                    </div>
                                </div>
                            {/if}
                        </div>

                        <!-- Accordion: Gateway Config -->
                        <div class="border border-[var(--border-color)] rounded-xl overflow-hidden bg-[var(--bg-panel)]/30">
                            <button 
                                type="button" 
                                onclick={() => collapsedSections.port = !collapsedSections.port} 
                                class="w-full flex items-center justify-between px-3.5 py-2.5 bg-[var(--bg-panel)] text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-secondary)] border-b border-[var(--border-color)] hover:bg-[var(--bg-hover)] transition"
                            >
                                <span>Gateway Config</span>
                                <span class="text-[9px] text-[var(--text-muted)] transition-transform duration-200 inline-block {collapsedSections.port ? '-rotate-90' : ''}">▼</span>
                            </button>
                            {#if !collapsedSections.port}
                                <div class="p-3.5 flex flex-col gap-2">
                                    <span class="text-[10px] font-bold text-[var(--text-secondary)]">Gateway Port</span>
                                    <input type="number" bind:value={serverPort} onchange={checkServerStatus} min="1" max="65535" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs font-mono focus:border-[var(--accent-color)] focus:outline-none transition" />
                                </div>
                            {/if}
                        </div>
                    </div>
                {:else}
                    <!-- MCP SERVERS PANE -->
                    <div class="flex flex-col gap-4">
                        {#if !isModelLoaded}
                            <div class="text-[11px] text-amber-600 dark:text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-xl p-3 leading-relaxed">
                                ⚠️ No model loaded. Load a model that supports tool calling to enable integrations.
                            </div>
                        {:else if !toolCallingSupported}
                            <div class="text-[11px] text-amber-600 dark:text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-xl p-3 leading-relaxed">
                                ⚠️ The loaded model ({activeModel}) does not support tool calling. Integrations are disabled.
                            </div>
                        {/if}

                        <div class="flex items-center justify-between border-b border-[var(--border-color)] pb-2.5 mb-1">
                            <span class="text-[11.5px] font-bold uppercase tracking-wider text-[var(--text-secondary)] select-none">Integrations</span>
                            <button type="button" disabled={!isModelLoaded || !toolCallingSupported} onclick={() => showAddMcpModal = true} class="px-2.5 py-1 bg-[var(--accent-color)] hover:bg-[var(--accent-hover)] disabled:bg-slate-300 dark:disabled:bg-slate-800 disabled:text-[var(--text-muted)] text-white text-[10px] font-bold rounded-lg transition active:scale-95 shadow-md shadow-black/10 flex items-center gap-1">
                                <span>+ Install</span>
                            </button>
                        </div>
                        
                        <!-- Search Box -->
                        <div class="relative">
                            <input type="text" disabled={!isModelLoaded || !toolCallingSupported} bind:value={mcpSearchQuery} placeholder="Filter plugins..." class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] disabled:opacity-50 rounded-lg text-[var(--text-primary)] placeholder-[var(--text-muted)] px-3 py-1.5 text-xs focus:border-[var(--accent-color)] focus:outline-none transition" />
                        </div>

                        <!-- Servers List -->
                        <div class="flex flex-col gap-2.5">
                            {#each filteredMcpServers as server}
                                <div class="border border-[var(--border-color)] rounded-xl overflow-hidden bg-[var(--bg-panel)]/30">
                                    <div class="flex items-center justify-between p-3 bg-[var(--bg-panel)]">
                                        <div class="flex items-center gap-2 max-w-[170px]">
                                            <!-- Toggle switch -->
                                            <button 
                                                type="button" 
                                                disabled={!isModelLoaded || !toolCallingSupported}
                                                onclick={() => toggleMcpServer(server.name, !server.enabled)} 
                                                class="relative inline-flex h-4 w-7.5 shrink-0 cursor-pointer rounded-full border border-slate-300/80 dark:border-slate-700 transition-colors duration-200 ease-in-out focus:outline-none {server.enabled ? 'bg-[var(--accent-color)] border-transparent' : 'bg-slate-300 dark:bg-slate-800'} {(!isModelLoaded || !toolCallingSupported) ? 'opacity-40 cursor-not-allowed' : ''}"
                                                role="switch"
                                                aria-checked={server.enabled}
                                                aria-label="Toggle MCP server"
                                            >
                                                <span class="pointer-events-none inline-block h-3 w-3 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out mt-[1px] ml-[1px] {server.enabled ? 'translate-x-3.5' : 'translate-x-0'}"></span>
                                            </button>
                                            <span class="text-xs font-bold text-[var(--text-primary)] truncate" title={server.name}>{server.name}</span>
                                        </div>
                                        
                                        <div class="flex items-center gap-2 select-none">
                                            {#if server.enabled}
                                                {#if server.connected}
                                                    <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" title="Connected"></span>
                                                {:else}
                                                    <span class="w-1.5 h-1.5 rounded-full bg-rose-500" title={server.error || "Failed to connect"}></span>
                                                {/if}
                                            {/if}
                                            <button type="button" title="Toggle Details" aria-label="Toggle Details" onclick={() => toggleServerExpanded(server.name)} class="p-1 hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] rounded transition">
                                                <span class="material-symbols-outlined text-[14px] transition-transform duration-150 {mcpExpandedServers[server.name] ? 'rotate-180' : ''}">expand_more</span>
                                            </button>
                                        </div>
                                    </div>

                                    {#if mcpExpandedServers[server.name]}
                                        <div class="p-3 border-t border-[var(--border-color)]/50 bg-[var(--bg-input)]/20 flex flex-col gap-2">
                                            {#if server.enabled}
                                                {#if server.connected}
                                                    <div class="flex items-center justify-between text-[10px] uppercase font-bold tracking-wider text-[var(--text-muted)] border-b border-[var(--border-color)]/30 pb-1 mb-1">
                                                        <span class="flex items-center gap-1">🛠️ Tools ({server.tools.length})</span>
                                                        <button type="button" disabled={!isModelLoaded || !toolCallingSupported} onclick={() => refreshMcpTools(server.name)} class="hover:text-[var(--text-primary)] transition disabled:opacity-40" title="Refresh tools">🔄</button>
                                                    </div>
                                                    {#if server.tools.length === 0}
                                                        <div class="text-[11px] text-[var(--text-muted)] italic">No tools found.</div>
                                                    {:else}
                                                        <div class="flex flex-col gap-1.5 max-h-[200px] overflow-y-auto pr-1">
                                                            {#each server.tools as tool}
                                                                {@const isToolEnabled = !server.disabled_tools.includes(tool.name)}
                                                                {@const isAllowed = server.allowed_tools && server.allowed_tools.includes(tool.name)}
                                                                <div class="flex items-center justify-between text-[11px] py-0.5">
                                                                    <label class="flex items-center gap-2 cursor-pointer text-[var(--text-secondary)] hover:text-[var(--text-primary)] select-none truncate flex-1 {(!isModelLoaded || !toolCallingSupported) ? 'pointer-events-none' : ''}">
                                                                        <input type="checkbox" disabled={!isModelLoaded || !toolCallingSupported} checked={isToolEnabled} onchange={() => toggleMcpTool(server.name, tool.name, !isToolEnabled)} class="rounded border-[var(--border-color)] bg-[var(--bg-input)] text-[var(--accent-color)] focus:ring-[var(--accent-color)] w-3.5 h-3.5 disabled:opacity-40" />
                                                                        <span class="truncate" title={tool.description}>{tool.name}</span>
                                                                    </label>
                                                                    {#if isAllowed}
                                                                        <button 
                                                                            type="button" 
                                                                            disabled={!isModelLoaded || !toolCallingSupported}
                                                                            onclick={() => toggleMcpToolPermission(server.name, tool.name, 'ask')} 
                                                                            class="text-[9.5px] text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/25 px-2 py-0.5 rounded-md font-mono select-none font-bold cursor-pointer transition active:scale-95 disabled:opacity-40 disabled:pointer-events-none"
                                                                            title="Click to require permission"
                                                                        >
                                                                            Allow
                                                                        </button>
                                                                    {:else}
                                                                        <button 
                                                                            type="button" 
                                                                            disabled={!isModelLoaded || !toolCallingSupported}
                                                                            onclick={() => toggleMcpToolPermission(server.name, tool.name, 'allow')} 
                                                                            class="text-[9.5px] text-amber-600 dark:text-amber-400 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/25 px-2 py-0.5 rounded-md font-mono select-none font-bold cursor-pointer transition active:scale-95 disabled:opacity-40 disabled:pointer-events-none"
                                                                            title="Click to allow automatically"
                                                                        >
                                                                            Ask
                                                                        </button>
                                                                    {/if}
                                                                </div>
                                                            {/each}
                                                        </div>
                                                    {/if}
                                                {:else}
                                                    <div class="text-[11px] text-rose-600 dark:text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-lg p-2 leading-relaxed break-all">
                                                        <strong>Error:</strong> {server.error || "Failed to start server process."}
                                                    </div>
                                                {/if}
                                            {:else}
                                                <div class="text-[11px] text-[var(--text-muted)] italic select-none py-1">Enable integration to view tools</div>
                                            {/if}
                                        </div>
                                    {/if}
                                </div>
                            {:else}
                                <div class="text-xs text-[var(--text-muted)] text-center py-6 select-none">No integrations found</div>
                            {/each}
                        </div>
                    </div>
                {/if}
            </div>
        </aside>
    {/if}

    <!-- ADD MCP SERVER MODAL -->
    {#if showAddMcpModal}
        <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 select-none animate-fade-in">
            <div class="bg-[var(--bg-panel)] border border-[var(--border-color)] w-full max-w-md rounded-2xl shadow-2xl overflow-hidden animate-scale-up">
                <!-- Modal Header -->
                <div class="p-5 border-b border-[var(--border-color)] flex items-center justify-between">
                    <h3 class="text-sm font-semibold text-[var(--text-primary)]">Install MCP Server</h3>
                    <button type="button" onclick={() => showAddMcpModal = false} class="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition">✕</button>
                </div>
                
                <!-- Modal Body -->
                <div class="p-5 flex flex-col gap-4">
                    <label class="flex flex-col gap-1.5">
                        <span class="text-[10px] font-bold text-[var(--text-secondary)]">Server ID (alphanumeric, no spaces)</span>
                        <input type="text" bind:value={newMcpName} placeholder="e.g. my-search-tool" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs focus:border-[var(--accent-color)] focus:outline-none transition" />
                    </label>
                    
                    <label class="flex flex-col gap-1.5">
                        <span class="text-[10px] font-bold text-[var(--text-secondary)]">Executable command</span>
                        <input type="text" bind:value={newMcpCommand} placeholder="e.g. npx, python3, node" class="w-full bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs focus:border-[var(--accent-color)] focus:outline-none transition" />
                    </label>
                    
                    <label class="flex flex-col gap-1.5">
                        <span class="text-[10px] font-bold text-[var(--text-secondary)]">Arguments (comma-separated or one per line)</span>
                        <textarea bind:value={newMcpArgs} placeholder="e.g. -y, @modelcontextprotocol/server-duckduckgo" class="w-full h-20 bg-[var(--bg-input)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] px-3 py-2 text-xs focus:border-[var(--accent-color)] focus:outline-none transition resize-none"></textarea>
                    </label>
                </div>
                
                <!-- Modal Footer -->
                <div class="p-5 border-t border-[var(--border-color)] bg-[var(--bg-input)]/40 flex justify-end gap-3">
                    <button type="button" onclick={() => showAddMcpModal = false} class="bg-transparent border border-[var(--border-color)] hover:border-[var(--text-secondary)] hover:bg-[var(--bg-hover)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] rounded-lg px-4 py-2 text-xs transition">
                        Cancel
                    </button>
                    <button type="button" onclick={installMcpServer} class="bg-[var(--accent-color)] hover:bg-[var(--accent-hover)] text-white rounded-lg px-5 py-2 text-xs font-medium transition active:scale-[0.98] shadow-md shadow-[var(--accent-color)]/10">
                        Install Integration
                    </button>
                </div>
            </div>
        </div>
    {/if}

</div>

<!-- SVELTE 5 RECURSIVE SNIPPETS -->
{#snippet renderFolder(folder, depth)}
    {#if folderHasMatches(folder.id)}
        <div class="flex flex-col gap-0.5 my-1" style="margin-left: {depth * 8}px">
            <!-- Folder header row -->
            <div class="group flex items-center justify-between rounded-lg px-2.5 py-1.5 text-xs font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors">
                <button type="button" onclick={() => { folder.isExpanded = !folder.isExpanded; saveToLocalStorage(); }} class="flex-1 flex items-center gap-2 text-left font-sans outline-none select-none">
                    <span class="material-symbols-outlined text-[12px] text-[var(--accent-color)] transition-transform duration-150 {folder.isExpanded ? 'rotate-90' : ''}">chevron_right</span>
                    <svg class="w-3.5 h-3.5 text-[var(--text-secondary)]" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-19.5 0A2.25 2.25 0 004.5 15h15a2.25 2.25 0 002.25-2.25m-19.5 0v.158c0 .882.365 1.722 1 2.302l1.62 1.458a2.25 2.25 0 001.5 1.571h8.22a2.25 2.25 0 001.5-.57l1.62-1.459a2.25 2.25 0 00.6-2.302V12.75m-19.5 0V7.5A2.25 2.25 0 014.5 5.25h5.053c.488 0 .954.19 1.302.53l1.378 1.378c.348.348.814.538 1.302.538h5.217A2.25 2.25 0 0121 9.964V12.75" />
                    </svg>
                    <span class="truncate">{folder.name}</span>
                </button>
                <div class="opacity-0 group-hover:opacity-100 flex gap-2 pl-2 transition-opacity duration-150 shrink-0 select-none">
                    <button type="button" onclick={(e) => createFolder(folder.id, e)} title="New Subfolder" class="hover:text-[var(--text-primary)] p-0.5">
                        <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                        </svg>
                    </button>
                    <button type="button" onclick={(e) => renameFolder(folder.id, e)} title="Rename Folder" class="hover:text-[var(--text-primary)] p-0.5">
                        <span class="material-symbols-outlined text-[12px]">edit</span>
                    </button>
                    <button type="button" onclick={(e) => moveFolder(folder.id, e)} title="Move Folder" class="hover:text-[var(--text-primary)] p-0.5">
                        <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M9 8.25H7.5a2.25 2.25 0 00-2.25 2.25v9a2.25 2.25 0 002.25 2.25h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25H15M9 12l3-3m0 0l3 3m-3-3v12" />
                        </svg>
                    </button>
                    <button type="button" onclick={(e) => deleteFolder(folder.id, e)} title="Delete Folder" class="hover:text-rose-500 p-0.5">
                        <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                        </svg>
                    </button>
                </div>
            </div>
            
            <!-- Subfolders and chats -->
            {#if folder.isExpanded}
                <div class="flex flex-col gap-0.5 border-l border-[var(--border-color)] ml-3.5 pl-1.5">
                    <!-- 1. Subfolders -->
                    {#each folders.filter(f => f.parentId === folder.id) as subfolder}
                        {@render renderFolder(subfolder, depth + 1)}
                    {/each}
                    
                    <!-- 2. Chats -->
                    {#each sessions.filter(s => s.folderId === folder.id && s.title.toLowerCase().includes(searchQuery.toLowerCase())) as session}
                        {@render renderChat(session, depth + 1)}
                    {/each}
 
                    <!-- Empty Placeholder -->
                    {#if folders.filter(f => f.parentId === folder.id).length === 0 && sessions.filter(s => s.folderId === folder.id && s.title.toLowerCase().includes(searchQuery.toLowerCase())).length === 0}
                        <div class="text-[10px] text-[var(--text-muted)] italic py-1 pl-4 select-none">Empty directory</div>
                    {/if}
                </div>
            {/if}
        </div>
    {/if}
{/snippet}

{#snippet renderChat(session, depth)}
    <div class="group flex items-center justify-between rounded-lg px-2.5 py-1.5 text-xs font-medium transition-all {session.id === currentChatId ? 'bg-[var(--accent-color)]/10 text-[var(--accent-color)] border border-[var(--accent-color)]/20 shadow-sm' : 'text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] border border-transparent'}">
        <button type="button" onclick={() => loadSession(session.id)} class="truncate flex-1 text-left flex items-center gap-2 outline-none select-none">
            <svg class="w-3.5 h-3.5 text-[var(--text-muted)] group-hover:text-[var(--accent-color)] transition-colors" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
            </svg>
            <span class="truncate">{session.title}</span>
        </button>
        <div class="opacity-0 group-hover:opacity-100 flex gap-2 pl-2 transition-opacity duration-150 shrink-0 select-none">
            <button type="button" onclick={(e) => renameSession(session.id, e)} title="Rename Chat" class="hover:text-[var(--text-primary)] p-0.5">
                <span class="material-symbols-outlined text-[12px]">edit</span>
            </button>
            <button type="button" onclick={(e) => moveSessionToFolder(session.id, e)} title="Move to Folder" class="hover:text-[var(--text-primary)] p-0.5">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M9 8.25H7.5a2.25 2.25 0 00-2.25 2.25v9a2.25 2.25 0 002.25 2.25h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25H15M9 12l3-3m0 0l3 3m-3-3v12" />
                </svg>
            </button>
            <button type="button" onclick={(e) => deleteSession(session.id, e)} title="Delete Chat" class="hover:text-rose-400 p-0.5">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                </svg>
            </button>
        </div>
    </div>
{/snippet}



<style>
    :global(:root) {
        --bg-main: #0c0d12;
        --bg-sidebar: #07080b;
        --bg-panel: #111218;
        --bg-input: #050508;
        --border-color: #1a1b23;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-color: #4f46e5;
        --accent-hover: #6366f1;
        --shadow-color: rgba(0, 0, 0, 0.4);
        --bg-hover: rgba(255, 255, 255, 0.04);
    }

    :global(.light-theme) {
        --bg-main: #ffffff;
        --bg-sidebar: #f8fafc;
        --bg-panel: #f1f5f9;
        --bg-input: #ffffff;
        --border-color: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --accent-color: #3b82f6;
        --accent-hover: #2563eb;
        --shadow-color: rgba(0, 0, 0, 0.05);
        --bg-hover: rgba(0, 0, 0, 0.04);
    }

    /* Premium Animations */
    .animate-spin {
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.2s ease-out forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .animate-scale-up {
        animation: scaleUp 0.18s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    }
    @keyframes scaleUp {
        from { transform: scale(0.96); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 99px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
</style>
