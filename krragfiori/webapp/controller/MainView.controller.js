sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], (Controller, JSONModel, MessageToast, MessageBox) => {
    "use strict";
 
    return Controller.extend("krragfiori.controller.MainView", {
        onInit() {
            var oChatModel = new JSONModel({
                messages: []
            });
            this.getView().setModel(oChatModel, "chatModel");
        },
        // --- FILE UPLOAD LOGIC ---
        onUploadPress: function () {
            var oFileUploader = this.byId("fileUploader");
            if (!oFileUploader.getValue()) {
                MessageToast.show("Please select a target file path first.");
                return;
            }
 
            sap.ui.core.BusyIndicator.show(0);
            // Executes dynamic multi-part data payload delivery to Python /api/v1/ingest via Approuter proxy
            oFileUploader.upload();
        },
 
        onUploadComplete: function (oEvent) {
            sap.ui.core.BusyIndicator.hide();
            var iStatus = oEvent.getParameter("status");
 
            if (iStatus === 200 || iStatus === 202) {
                MessageBox.success("Document dispatched successfully. Background vector processing initialized.");
                this.byId("fileUploader").setValue("");
            } else {
                MessageBox.error("Document ingestion pipeline rejection. Status code returned: " + iStatus);
            }
        },
 
        // --- LIVE COPILOT CONVERSATION LOGIC (UPDATED FOR FEEDINPUT) ---
        onSendPrompt: function (oEvent) {
            // 1. sap.m.FeedInput automatically provides the typed text inside the "value" parameter
            var sQuery = oEvent.getParameter("value");
            if (!sQuery || !sQuery.trim()) { return; }
 
            var oModel = this.getView().getModel("chatModel");
            var aMessages = oModel.getProperty("/messages");
            var sTime = new Date().toLocaleTimeString();
 
            // 2. Append the User's Question to the Feed panel history array
            aMessages.push({
                sender: "You",
                text: sQuery,
                timestamp: sTime
            });
            oModel.setProperty("/messages", aMessages);
 
            // 3. Append a placeholder "AI Thinking..." log entry
            aMessages.push({
                sender: "AI Copilot",
                text: "Thinking...",
                timestamp: sTime
            });
            oModel.setProperty("/messages", aMessages);
 
            // Track the index position of the AI message so we can overwrite it when the backend responds
            var iAiItemIndex = aMessages.length - 1;
 
            // 4. Trigger Fetch Execution targeting your local Python Debugger or BTP route
            fetch("/api/v1/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: sQuery })
            })
                .then(function (response) {
                    return response.json();
                })
                .then(function (data) {
                    // Update the "Thinking..." text placeholder with the actual answer string from Python
                    aMessages[iAiItemIndex].text = data.answer || "No response text payload returned from model.";
                    oModel.setProperty("/messages", aMessages);
                })
                .catch(function (error) {
                    // Overwrite the placeholder with a clean connection failure alert message
                    aMessages[iAiItemIndex].text = "Error reaching the AI vector processing pipeline server.";
                    oModel.setProperty("/messages", aMessages);
                    console.error("RAG Pipeline Route Connection Dropped:", error);
                });
        }
 
    });
});
 