import os

with open('server/server.py', 'r') as f:
    content = f.read()

sse_client_code = """
class SSEMCPServerClient:
    def __init__(self, name, url, headers=None):
        self.name = name
        self.url = url
        self.headers = headers or {}
        self.post_url = None
        self.request_id = 1
        self.pending_requests = {}
        self.tools = []
        self.connected = False
        self.error = None
        self.read_thread = None
        self._stop_event = __import__('threading').Event()

    def start(self):
        try:
            self.read_thread = __import__('threading').Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            
            # Wait for endpoint to be received
            import time
            start_time = time.time()
            while not self.post_url and time.time() - start_time < 5:
                time.sleep(0.1)
                
            if not self.post_url:
                raise Exception("Did not receive endpoint from SSE stream in time.")
                
            self.connected = True
            self.initialize()
            self.fetch_tools()
        except Exception as e:
            self.connected = False
            self.error = str(e)
            print(f"Error starting SSE MCP server {self.name}: {e}")

    def _read_loop(self):
        import requests
        import urllib.parse
        import json
        try:
            req_headers = self.headers.copy()
            req_headers["Accept"] = "text/event-stream"
            with requests.get(self.url, headers=req_headers, stream=True, timeout=10) as resp:
                resp.raise_for_status()
                
                event_type = None
                for line in resp.iter_lines(decode_unicode=True):
                    if self._stop_event.is_set():
                        break
                    if line is None:
                        continue
                    if line == "":
                        event_type = None
                        continue
                        
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                    elif line.startswith("data: "):
                        data = line[6:]
                        if event_type == "endpoint":
                            self.post_url = urllib.parse.urljoin(self.url, data)
                        elif event_type == "message":
                            try:
                                msg = json.loads(data)
                                msg_id = msg.get('id')
                                if msg_id is not None:
                                    store = self.pending_requests.get(msg_id)
                                    if store:
                                        store['response'] = msg
                                        store['event'].set()
                            except Exception:
                                pass
        except Exception as e:
            print(f"SSE read loop exception for {self.name}: {e}")
        finally:
            self.connected = False

    def send_request(self, method, params, timeout=10):
        if not self.connected or not self.post_url:
            return {"error": "Server not connected or missing endpoint"}
            
        current_id = self.request_id
        self.request_id += 1
        
        event = __import__('threading').Event()
        req_store = {"event": event, "response": None}
        self.pending_requests[current_id] = req_store
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": current_id
        }
        
        import requests
        try:
            resp = requests.post(self.post_url, json=payload, headers=self.headers, timeout=timeout)
            resp.raise_for_status()
        except Exception as e:
            self.pending_requests.pop(current_id, None)
            return {"error": f"POST failed: {e}"}
            
        success = event.wait(timeout=timeout)
        self.pending_requests.pop(current_id, None)
        
        if not success:
            return {"error": "Request timed out"}
            
        return req_store["response"]

    def initialize(self):
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "unified-mlx-client", "version": "1.0.0"}
        }
        res = self.send_request("initialize", params)
        if res and "error" not in res:
            # Send initialized notification
            notify = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            import requests
            try:
                requests.post(self.post_url, json=notify, headers=self.headers, timeout=5)
            except Exception:
                pass

    def fetch_tools(self):
        res = self.send_request("tools/list", {})
        if res and "result" in res and "tools" in res["result"]:
            self.tools = res["result"]["tools"]
        else:
            self.tools = []

    def call_tool(self, tool_name, arguments):
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        return self.send_request("tools/call", params, timeout=30)

    def stop(self):
        self.connected = False
        self._stop_event.set()

def sync_mcp_servers():
"""

content = content.replace("def sync_mcp_servers():\n", sse_client_code)

old_start_loop = """
    # 2. Start newly enabled servers
    for name, cfg in servers.items():
        if cfg.get("enabled", False) and name not in mcp_clients:
            if "command" not in cfg:
                print(f"Skipping MCP server '{name}': No 'command' found in config (only stdio transport is supported).")
                continue
            print(f"Starting MCP server: {name}")
            client = MCPServerClient(name, cfg["command"], cfg.get("args", []), cfg.get("env"))
            client.start()
            mcp_clients[name] = client
"""

new_start_loop = """
    # 2. Start newly enabled servers
    for name, cfg in servers.items():
        if cfg.get("enabled", False) and name not in mcp_clients:
            print(f"Starting MCP server: {name}")
            if "url" in cfg:
                # SSE transport
                client = SSEMCPServerClient(name, cfg["url"], cfg.get("headers", {}))
                client.start()
                mcp_clients[name] = client
            elif "command" in cfg:
                # Stdio transport
                client = MCPServerClient(name, cfg["command"], cfg.get("args", []), cfg.get("env"))
                client.start()
                mcp_clients[name] = client
            else:
                print(f"Skipping MCP server '{name}': Neither 'command' nor 'url' found in config.")
"""

content = content.replace(old_start_loop.strip(), new_start_loop.strip())

with open('server/server.py', 'w') as f:
    f.write(content)

