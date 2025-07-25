# OpenWebUI 工具套件 简体中文版说明  
---

## 1. 背景与目标  
默认的 OpenWebUI 模型很“被动”：  
- 每次都得手动选工具  
- 搜索结果常常不给力  

本项目给本地模型装上“自动驾驶”：  
- 两层智能路由，自动决定要不要搜索、画图、跑代码 
- 体验直接对标 ChatGPT / Gemini 的联网、画图、代码执行能力
- 未来添加跟多工具的能力

---

## 2. 功能速览  
| 功能 | 一句话说明 |
|---|---|
| **自动路由** | 主路由拦截每条消息，自动挑工具，无需手动点选 |
| **三种搜索模式** | 用 Exa.ai 实现：  
• **Crawl** 直接读指定网页  
• **Quick** 5 秒内给摘要答案  
• **Complete** 深度研究，多轮搜索+总结 |
| **智能画图** | 自动优化提示词，生成更高质量图片，并能把图片信息再喂回对话 |
| **代码解释器** | 支持原生解释器，也支持 Jupyter，可返回文件下载链接 |
| **实时状态** | 每步操作都有提示，不让你干等 |

https://private-user-images.githubusercontent.com/212266166/468440945-696cb316-c160-4210-a0dc-f87a04be1647.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTM0NzIwOTEsIm5iZiI6MTc1MzQ3MTc5MSwicGF0aCI6Ii8yMTIyNjYxNjYvNDY4NDQwOTQ1LTY5NmNiMzE2LWMxNjAtNDIxMC1hMGRjLWY4N2EwNGJlMTY0Ny5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNzI1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDcyNVQxOTI5NTFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kZWJjNzZlOGRhYzVmMDhmYjI4MjdhMDdkNzhjMzE0NTE4NzMyYmJmY2JiNTAzNDc4NmYzYjFkNWViZDlhMDA2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.kUOVvhBjm64d2XFGPbsCGy5s6QMwuShsEoGlPEVqTs4

---

## 3. 配置：Valves 与模型选择  
所有工具都能通过“Valves”微调，平衡速度、成本、智能。

### 3.1 主路由 Auto Tool Selector  
| Valve | 作用 | 推荐 |
|---|---|---|
| `helper_model` | 决定用不用工具 | GPT-4.1-mini / Llama3-8B |
| `vision_model` | 看图说话 | GPT-4.1 / Gemini 2.0 Flash |
| `vision_injection_models` | 需要图片注解的非识图模型| |
| `history_char_limit` | 保留多少历史字符，省 token | 默认即可 |

### 3.2 搜索路由 Exa Search Router  
| Valve | 作用 | 推荐 |
|---|---|---|
| `exa_api_key` | **必填** Exa 的 API Key | 去 exa.ai 申请 |
| `router_model` | 选哪种搜索模式 | GPT-4.1-mini |
| `quick_search_model` | Quick 模式用 | GPT-4.1-mini / Gemini 2.0 Flash |
| `complete_agent_model` | Complete 模式“大脑” | GPT-4.1 / Gemini 2.0 Flash / Claude 4 Sonnet |
| `complete_summarizer_model` | 最终总结 (long context) | Gemini 2.5 Flash / Llama 4 Maverick  |

#### 搜索深度参数（可默认）
- Quick 模式：`quick_urls_to_search`、`quick_queries_to_crawl`、`quick_max_context_chars`
- Complete 模式：`complete_urls_to_search_per_query`、`complete_queries_to_crawl`、`complete_queries_to_generate`、`complete_max_search_iterations`

---

## 4. 安装与使用

### Step 0：选文件  
- 用 **原生解释器** → 选 `auto_tool_selector_default_code_interpter.py`  
- 用 **Jupyter** → 选 `auto_tool_selector_jupyter_ci.py`

### Step 1（仅 Jupyter 用户）：放 `uploader.py`  
把 `uploader.py` 扔进 Jupyter 启动的根目录，否则 AI 无法返回文件下载链接。

### Step 2：装依赖  
在宿主机执行一次即可：  
    docker exec -it open-webui bash  
    pip install exa_py  
    exit  
    docker restart open-webui  

### Step 3：后台配置  
1. **新建 Function（主路由）**  
   Admin Settings → Functions → New Function → 粘贴对应 `Auto Tool Selector` 文件  
2. **新建 Tool（搜索路由）**  
   Workspace → Tools → New Tool → 粘贴 `Exa Search Router` → **Tool ID 必须填 `exa_router_search`**  
3. **填 API Key**  
   在 `Exa Search Router` 的 Valves 里填 `exa_api_key`

### Step 4：启用  
- 只在想增强的模型里 **勾选 `Auto Tool Selector` 这一个 Function**  
- **不要** 直接勾选 `Exa Search Router`，否则主路由会失效

---

## 5. 常见问题 FAQ  
**Q1：为什么要手动装 `exa_py`？**  
OpenWebUI 不会自动装库，Requirements 只是注释，必须进容器手动装。

**Q2：Function 和 Tool 有啥区别？**  
- Function（过滤器）：在模型回答前运行，负责“改道”。  
- Tool：模型回答过程中可调用的工具。

**Q3：为什么复杂问题没触发 Complete 搜索？**  
Complete 很耗资源，有时候你的明确说“深度研究”之类关键词时才触发。

---

## 6. 流程图  
![架构图](https://github.com/user-attachments/assets/e79f7658-020f-4804-8d16-e4414ad781e8)

---


