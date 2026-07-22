"""
小智Pro 手势自然语言解析器

小智Pro 内置手势识别，输出自然语言描述（如"用户比出了两指朝上的手势"）。
本模块提供工具函数，交给 LLM 理解任意手势措辞，提取五维方向和手指数。
不做硬编码模板匹配。

用法：
    from xiaozhi_gesture_parser import is_camera_control, build_gesture_prompt, parse_llm_result
"""

# ============================================================
# 摄像头控制
# ============================================================

CAMERA_ON = ["打开摄像头", "开启摄像头", "启动摄像头", "开摄像头"]
CAMERA_OFF = ["关闭摄像头", "关掉摄像头", "关摄像头", "停止摄像头"]


def is_camera_control(text: str) -> str | None:
    """判断是否是摄像头控制指令。返回 "open" / "close" / None。"""
    for kw in CAMERA_ON:
        if kw in text:
            return "open"
    for kw in CAMERA_OFF:
        if kw in text:
            return "close"
    return None


# ============================================================
# LLM Prompt 构建
# ============================================================

GESTURE_SYSTEM_PROMPT = """你是手势意图解析器。用户做手势，小智Pro 用自然语言描述手势，你需要提取方向和手指数。

方向映射：
- 朝上/向上/上方/往上 → UP（上报给上级）
- 朝下/向下/下方/往下 → DOWN（委派给下属）
- 朝左/向左/左侧/往左 → LEFT（存档任务）
- 朝右/向右/右侧/往右 → RIGHT（Agent分派）
- 握拳/拳头/攥拳/捏拳 → CENTER（确认）

手指数映射：
- 食指/一根手指/一个手指/单指 → 1
- 两指/两根手指/两个手指/双指 → 2
- 三指/三根手指 → 3
- 四指/四根手指 → 4
- 握拳 → 0

只输出一个 JSON 对象，不要其他内容：
{"direction":"UP","finger_count":2}

如果文本不是手势描述，输出：
{"direction":null,"finger_count":0}"""


def build_gesture_prompt(user_text: str) -> str:
    """构造 LLM prompt，让 LLM 理解手势自然语言描述。"""
    return GESTURE_SYSTEM_PROMPT + f"\n\n用户输入：{user_text}"


# ============================================================
# LLM 结果解析
# ============================================================

def parse_llm_result(llm_output: str) -> dict | None:
    """
    解析 LLM 返回的 JSON，提取 direction 和 finger_count。
    返回 {"direction":"UP","finger_count":2} 或 None（不是手势描述）。
    """
    import json
    try:
        # 尝试从 LLM 输出中提取 JSON
        text = llm_output.strip()
        # 去掉可能的 markdown 代码块包裹
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(text)
        if result.get("direction"):
            return {
                "direction": result["direction"],
                "finger_count": result.get("finger_count", 0),
            }
    except (json.JSONDecodeError, KeyError):
        pass
    return None
