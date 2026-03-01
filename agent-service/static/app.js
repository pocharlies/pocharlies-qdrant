/* ============================================================
   Pocharlies Agent — Chat UI (WebSocket client)
   ============================================================
   SECURITY: This file never uses innerHTML, outerHTML,
   insertAdjacentHTML, or similar unsafe DOM methods.
   All content is set via textContent / createElement +
   appendChild only.
   ============================================================ */

"use strict";

// -----------------------------------------------------------
// State
// -----------------------------------------------------------

var state = {
    ws: null,
    threadId: null,
    threads: [],            // [{id, name, lastMessage}]
    threadHistory: {},      // threadId → [{type, role, content, ...}]
    connected: false,
    currentAssistantEl: null,
};

// -----------------------------------------------------------
// DOM references
// -----------------------------------------------------------

var $messages     = document.getElementById("messages");
var $input        = document.getElementById("message-input");
var $sendBtn      = document.getElementById("send-btn");
var $newThreadBtn = document.getElementById("new-thread-btn");
var $threadList   = document.getElementById("thread-list");
var $statusDot    = document.getElementById("status-indicator");
var $statusText   = document.getElementById("status-text");
var $helpBtn      = document.getElementById("help-btn");
var $helpOverlay  = document.getElementById("help-overlay");
var $helpPopup    = document.getElementById("help-popup");
var $helpCloseBtn = document.getElementById("help-close-btn");

// -----------------------------------------------------------
// Utilities
// -----------------------------------------------------------

function generateId() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
        var r = (Math.random() * 16) | 0;
        var v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

function timestamp() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
    requestAnimationFrame(function () {
        $messages.scrollTop = $messages.scrollHeight;
    });
}

// -----------------------------------------------------------
// LocalStorage persistence
// -----------------------------------------------------------

var STORAGE_KEY = "pocharlies_threads";

function saveToStorage() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            threads: state.threads,
            history: state.threadHistory,
            activeThread: state.threadId,
        }));
    } catch (e) {
        // Storage full or unavailable
    }
}

function loadFromStorage() {
    try {
        var raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return false;
        var data = JSON.parse(raw);
        if (data.threads && Array.isArray(data.threads)) {
            state.threads = data.threads;
            state.threadHistory = data.history || {};
            state.threadId = data.activeThread || null;
            return true;
        }
    } catch (e) {
        // Corrupt data — start fresh
    }
    return false;
}

// -----------------------------------------------------------
// Status
// -----------------------------------------------------------

function updateStatus(connected) {
    state.connected = connected;
    if (connected) {
        $statusDot.classList.add("connected");
        $statusDot.classList.remove("disconnected");
        $statusText.textContent = "Connected";
    } else {
        $statusDot.classList.remove("connected");
        $statusDot.classList.add("disconnected");
        $statusText.textContent = "Disconnected";
    }
}

// -----------------------------------------------------------
// WebSocket
// -----------------------------------------------------------

function connect() {
    var protocol = location.protocol === "https:" ? "wss:" : "ws:";
    var url = protocol + "//" + location.host + "/api/chat/ws";

    var ws = new WebSocket(url);
    state.ws = ws;

    ws.addEventListener("open", function () {
        updateStatus(true);
    });

    ws.addEventListener("close", function () {
        updateStatus(false);
        state.ws = null;
        setTimeout(connect, 3000);
    });

    ws.addEventListener("error", function (err) {
        console.error("[ws] error:", err);
    });

    ws.addEventListener("message", function (event) {
        var data;
        try {
            data = JSON.parse(event.data);
        } catch (e) {
            console.error("[ws] bad JSON:", event.data);
            return;
        }
        handleServerMessage(data);
    });
}

// -----------------------------------------------------------
// Server message handler
// -----------------------------------------------------------

function handleServerMessage(data) {
    switch (data.type) {
        case "token":
            if (!state.currentAssistantEl) {
                startStreaming();
            }
            appendToken(data.content);
            break;

        case "tool_call":
            addToolCall(data.name, data.args);
            break;

        case "tool_result":
            updateToolResult(data.name, data.content);
            break;

        case "done":
            finalizeStreaming();
            if (data.thread_id) {
                state.threadId = data.thread_id;
                updateCurrentThread();
            }
            break;

        case "error":
            addMessage("error", data.message || "Unknown error");
            finalizeStreaming();
            break;

        case "interrupt":
            addInterrupt(data);
            break;

        default:
            console.warn("[ws] unknown message type:", data.type);
    }
}

// -----------------------------------------------------------
// History helpers — record every event in the data model
// -----------------------------------------------------------

function getHistory() {
    if (!state.threadId) return [];
    if (!state.threadHistory[state.threadId]) {
        state.threadHistory[state.threadId] = [];
    }
    return state.threadHistory[state.threadId];
}

function pushRecord(record) {
    getHistory().push(record);
    saveToStorage();
}

// -----------------------------------------------------------
// Messages
// -----------------------------------------------------------

function addMessage(role, content) {
    var ts = timestamp();
    pushRecord({ type: "message", role: role, content: content, time: ts });
    renderMessage(role, content, ts);
}

function renderMessage(role, content, ts) {
    var el = document.createElement("div");
    el.classList.add("message", role);

    var textEl = document.createElement("span");
    textEl.classList.add("message-text");
    textEl.textContent = content;
    el.appendChild(textEl);

    var tsEl = document.createElement("span");
    tsEl.classList.add("message-timestamp");
    tsEl.textContent = ts;
    el.appendChild(tsEl);

    $messages.appendChild(el);
    scrollToBottom();
    return el;
}

// -----------------------------------------------------------
// Streaming
// -----------------------------------------------------------

function startStreaming() {
    var el = document.createElement("div");
    el.classList.add("message", "assistant");

    var textEl = document.createElement("span");
    textEl.classList.add("message-text");
    el.appendChild(textEl);

    var tsEl = document.createElement("span");
    tsEl.classList.add("message-timestamp");
    tsEl.textContent = timestamp();
    el.appendChild(tsEl);

    $messages.appendChild(el);
    state.currentAssistantEl = textEl;
    scrollToBottom();
}

function appendToken(token) {
    if (state.currentAssistantEl) {
        state.currentAssistantEl.textContent += token;
        scrollToBottom();
    }
}

function finalizeStreaming() {
    if (state.currentAssistantEl) {
        var content = state.currentAssistantEl.textContent;
        var tsEl = state.currentAssistantEl.parentElement
            ? state.currentAssistantEl.parentElement.querySelector(".message-timestamp")
            : null;
        var timeStr = tsEl ? tsEl.textContent : timestamp();

        if (content) {
            pushRecord({ type: "message", role: "assistant", content: content, time: timeStr });

            var thread = state.threads.find(function (t) { return t.id === state.threadId; });
            if (thread) {
                thread.lastMessage = content.slice(0, 80);
                renderThreads();
                saveToStorage();
            }
        }
    }
    state.currentAssistantEl = null;
}

// -----------------------------------------------------------
// Tool calls
// -----------------------------------------------------------

function addToolCall(name, args) {
    var argsStr = typeof args === "string" ? args : JSON.stringify(args, null, 2);
    pushRecord({ type: "tool_call", name: name, args: argsStr });
    renderToolCall(name, argsStr);
}

function renderToolCall(name, argsStr) {
    var el = document.createElement("div");
    el.classList.add("tool-call");
    el.setAttribute("data-tool-name", name);

    var header = document.createElement("div");
    header.classList.add("tool-call-header");

    var icon = document.createElement("span");
    icon.classList.add("tool-call-icon");
    icon.textContent = "\u2699";
    header.appendChild(icon);

    var nameEl = document.createElement("span");
    nameEl.classList.add("tool-call-name");
    nameEl.textContent = name;
    header.appendChild(nameEl);

    var toggle = document.createElement("span");
    toggle.classList.add("tool-call-toggle");
    toggle.textContent = "\u25B6";
    header.appendChild(toggle);

    el.appendChild(header);

    var body = document.createElement("div");
    body.classList.add("tool-call-body");

    var argsLabel = document.createElement("div");
    argsLabel.classList.add("tool-call-label");
    argsLabel.textContent = "Arguments";
    body.appendChild(argsLabel);

    var argsContent = document.createElement("pre");
    argsContent.classList.add("tool-call-content");
    argsContent.textContent = argsStr;
    body.appendChild(argsContent);

    el.appendChild(body);

    header.addEventListener("click", function () {
        el.classList.toggle("expanded");
    });

    $messages.appendChild(el);
    scrollToBottom();
}

function updateToolResult(name, content) {
    pushRecord({ type: "tool_result", name: name, content: content });
    renderToolResult(name, content);
}

function renderToolResult(name, content) {
    var toolCalls = $messages.querySelectorAll('.tool-call[data-tool-name="' + CSS.escape(name) + '"]');
    var target = toolCalls.length > 0 ? toolCalls[toolCalls.length - 1] : null;

    if (!target) {
        renderToolCall(name, "");
        toolCalls = $messages.querySelectorAll('.tool-call[data-tool-name="' + CSS.escape(name) + '"]');
        target = toolCalls[toolCalls.length - 1];
    }

    var body = target.querySelector(".tool-call-body");
    if (!body) return;

    var resultDiv = document.createElement("div");
    resultDiv.classList.add("tool-call-result");

    var label = document.createElement("div");
    label.classList.add("tool-call-label");
    label.textContent = "Result";
    resultDiv.appendChild(label);

    var pre = document.createElement("pre");
    pre.classList.add("tool-call-content");
    pre.textContent = content;
    resultDiv.appendChild(pre);

    body.appendChild(resultDiv);
    target.classList.add("expanded");
    scrollToBottom();
}

// -----------------------------------------------------------
// Interrupt (approve / reject)
// -----------------------------------------------------------

function addInterrupt(data) {
    var msg = data.message || "The agent is requesting approval to proceed.";
    pushRecord({ type: "interrupt", message: msg, interrupt_id: data.interrupt_id || null });
    renderInterrupt(msg, data.interrupt_id || null, false);
}

function renderInterrupt(message, interruptId, resolved) {
    var el = document.createElement("div");
    el.classList.add("interrupt");

    var text = document.createElement("div");
    text.classList.add("interrupt-text");
    text.textContent = message;
    el.appendChild(text);

    if (!resolved) {
        var buttons = document.createElement("div");
        buttons.classList.add("interrupt-buttons");

        var approveBtn = document.createElement("button");
        approveBtn.classList.add("btn-approve");
        approveBtn.textContent = "Approve";
        approveBtn.addEventListener("click", function () {
            sendInterruptResponse("approve", interruptId);
            approveBtn.disabled = true;
            rejectBtn.disabled = true;
            text.textContent += " [Approved]";
        });
        buttons.appendChild(approveBtn);

        var rejectBtn = document.createElement("button");
        rejectBtn.classList.add("btn-reject");
        rejectBtn.textContent = "Reject";
        rejectBtn.addEventListener("click", function () {
            sendInterruptResponse("reject", interruptId);
            approveBtn.disabled = true;
            rejectBtn.disabled = true;
            text.textContent += " [Rejected]";
        });
        buttons.appendChild(rejectBtn);

        el.appendChild(buttons);
    }

    $messages.appendChild(el);
    scrollToBottom();
}

function sendInterruptResponse(action, interruptId) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    state.ws.send(JSON.stringify({
        type: "interrupt_response",
        action: action,
        thread_id: state.threadId,
        interrupt_id: interruptId,
    }));
}

// -----------------------------------------------------------
// Help popup
// -----------------------------------------------------------

function openHelp() {
    if ($helpOverlay) $helpOverlay.classList.remove("hidden");
    if ($helpPopup) $helpPopup.classList.remove("hidden");
}

function closeHelp() {
    if ($helpOverlay) $helpOverlay.classList.add("hidden");
    if ($helpPopup) $helpPopup.classList.add("hidden");
}

// -----------------------------------------------------------
// Send user message
// -----------------------------------------------------------

function send() {
    var content = $input.value.trim();
    if (!content) return;
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;

    closeHelp();

    if (!state.threadId) {
        newThread(content);
    }

    addMessage("user", content);

    var thread = state.threads.find(function (t) { return t.id === state.threadId; });
    if (thread) {
        thread.lastMessage = content.slice(0, 80);
        if (thread.name === "New thread") {
            thread.name = content.slice(0, 40);
        }
        renderThreads();
        saveToStorage();
    }

    state.ws.send(JSON.stringify({
        type: "message",
        content: content,
        thread_id: state.threadId,
    }));

    $input.value = "";
    autoResize();
}

// -----------------------------------------------------------
// Thread management (data-model based)
// -----------------------------------------------------------

function clearMessages() {
    while ($messages.firstChild) {
        $messages.removeChild($messages.firstChild);
    }
}

function renderThreadHistory(threadId) {
    clearMessages();
    var history = state.threadHistory[threadId];
    if (!history || history.length === 0) return;

    for (var i = 0; i < history.length; i++) {
        var rec = history[i];
        switch (rec.type) {
            case "message":
                renderMessage(rec.role, rec.content, rec.time);
                break;
            case "tool_call":
                renderToolCall(rec.name, rec.args);
                break;
            case "tool_result":
                renderToolResult(rec.name, rec.content);
                break;
            case "interrupt":
                renderInterrupt(rec.message, rec.interrupt_id, true);
                break;
        }
    }
    scrollToBottom();
}

function newThread(firstMessage) {
    var id = generateId();
    var name = firstMessage ? firstMessage.slice(0, 40) : "New thread";
    state.threadId = id;
    state.currentAssistantEl = null;
    state.threads.unshift({ id: id, name: name, lastMessage: "" });
    state.threadHistory[id] = [];
    clearMessages();
    renderThreads();
    saveToStorage();
    $input.focus();
}

function switchThread(id) {
    if (state.threadId === id) return;
    state.threadId = id;
    state.currentAssistantEl = null;
    renderThreadHistory(id);
    renderThreads();
    saveToStorage();
    $input.focus();
}

function updateCurrentThread() {
    var exists = state.threads.some(function (t) { return t.id === state.threadId; });
    if (!exists && state.threadId) {
        state.threads.unshift({
            id: state.threadId,
            name: "Thread",
            lastMessage: "",
        });
        renderThreads();
        saveToStorage();
    }
}

function renderThreads() {
    while ($threadList.firstChild) {
        $threadList.removeChild($threadList.firstChild);
    }

    state.threads.forEach(function (thread) {
        var btn = document.createElement("button");
        btn.classList.add("thread-item");
        if (thread.id === state.threadId) {
            btn.classList.add("active");
        }

        var nameSpan = document.createElement("span");
        nameSpan.classList.add("thread-name");
        nameSpan.textContent = thread.name;
        btn.appendChild(nameSpan);

        var previewSpan = document.createElement("span");
        previewSpan.classList.add("thread-preview");
        previewSpan.textContent = thread.lastMessage || "";
        btn.appendChild(previewSpan);

        btn.addEventListener("click", function () {
            switchThread(thread.id);
        });

        $threadList.appendChild(btn);
    });
}

// -----------------------------------------------------------
// Auto-resize textarea
// -----------------------------------------------------------

function autoResize() {
    $input.style.height = "auto";
    $input.style.height = Math.min($input.scrollHeight, 200) + "px";
}

// -----------------------------------------------------------
// Event listeners
// -----------------------------------------------------------

$input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
    }
});

$input.addEventListener("input", autoResize);
$sendBtn.addEventListener("click", send);
$newThreadBtn.addEventListener("click", function () { newThread(); });

if ($helpBtn) $helpBtn.addEventListener("click", openHelp);
if ($helpCloseBtn) $helpCloseBtn.addEventListener("click", closeHelp);
if ($helpOverlay) $helpOverlay.addEventListener("click", closeHelp);

document.querySelectorAll(".help-action").forEach(function (btn) {
    btn.addEventListener("click", function () {
        var prompt = btn.getAttribute("data-prompt");
        if (prompt) {
            closeHelp();
            $input.value = prompt;
            send();
        }
    });
});

// -----------------------------------------------------------
// Initialise
// -----------------------------------------------------------

(function init() {
    var restored = loadFromStorage();
    if (restored && state.threads.length > 0) {
        if (state.threadId) {
            renderThreadHistory(state.threadId);
        } else {
            state.threadId = state.threads[0].id;
            renderThreadHistory(state.threadId);
        }
        renderThreads();
    } else {
        newThread();
    }
    connect();
})();
