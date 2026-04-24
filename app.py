import os
from flask import Flask, render_template, request, jsonify
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = anthropic.Anthropic()

# ── 共用語氣設定 ──────────────────────────────────────────────
TONES = {
    "溫暖鼓勵": "語氣溫暖親切，充滿鼓勵與肯定，讓人感受到老師的關懷",
    "正式嚴謹": "語氣正式、嚴謹客觀，措辭專業得體",
    "親切友善": "語氣輕鬆自然，像朋友般親切，拉近師生（或師親）距離",
    "嚴肅提醒": "語氣較為嚴肅，帶有提醒與期許意味，督促對方努力",
}

# ── 學校老師評語：評語重點 ────────────────────────────────────
SCHOOL_GUIDES = {
    "學習態度": "學習態度、專注度、學習熱忱與主動性",
    "學業進步": "學業成績表現、進步幅度、理解與吸收能力",
    "待加強項目": "需要改善的地方、具體努力方向與建議",
    "品德表現": "品德行為、禮貌待人、與同儕互動情形",
    "整體表現": "學習與生活各方面的整體綜合表現",
    "特殊才能": "特殊能力、才藝展現、獨特的天賦或潛力",
}

# ── 農耕側記：年級 ────────────────────────────────────────────
GRADES = [
    "小一", "小二", "小三", "小四", "小五", "小六",
    "國一", "國二", "國三",
]

# ── 禁用詞規則（農耕風格側記共用）────────────────────────────
FORBIDDEN_WORDS_NOTE = """
【禁止用語】以下詞語禁止出現，改用更口語的說法：
- 「透過本次活動」→ 今天／這堂課
- 「深刻體會」→ 真的感受到
- 「寓教於樂」→ 直接描述就好
- 「相信孩子們」→ 直接說孩子做了什麼
- 「習得」→ 學到／知道了
- 「培養學生」→ 孩子們
- 「此次課程」→ 今天
- 「展現出」→ 直接描述行為
- 「莫大的收穫」→ 真的學到東西了
"""

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _student_ref(gender: str, name: str) -> tuple:
    pronoun = "他" if gender == "男" else "她"
    noun = name.strip() if name and name.strip() else "小朋友"
    return noun, pronoun


# ─────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────

def build_farming_prompt(keywords: str, grade: str, gender: str = "男", name: str = "") -> str:
    return f"""你是一位國小／國中自然科老師的文字助手，負責將農耕課關鍵字轉化為給家長看的課程側記。

【年級】{grade}
【關鍵字】{keywords}

【文章固定結構】
1. 標題：農耕課課程側記 ＋ 1 個 emoji（寫在文章最前面）
2. 首段（教育內容）：簡單介紹今日主題的農業或科學原理，口語自然，約 40 字
3. 中段（課程過程）：描述孩子實際操作的畫面，活潑有趣，像在說故事，約 80–100 字
4. 末段（溫馨結尾）：一句有溫度的收尾，讓家長感受農耕課的價值，約 20–30 字

全文總字數：150–200 字

【語氣規則】
- 口語自然，像老師在跟家長群組聊天的感覺
- 可用破折號——製造節奏感
- 可用驚嘆號表現孩子的活力！
- Emoji 全文使用 2–4 個，分散在不同段落，不集中

{FORBIDDEN_WORDS_NOTE}

【參考範文語氣基準】
農耕課課程側記 🌱

鬆土能讓土壤透氣、幫助根系呼吸，也讓水分和養分更容易被作物吸收——
看起來簡單，其實是讓植物長好的關鍵第一步 💪

今天國一的孩子們一進菜圃，就捲起袖子開始鬆土！
鬆完土，大家合力堆起地瓜專屬的土堆，有模有樣！
接著迎來今天最開心的時刻——拔紅蘿蔔🥕
第一次採收，孩子們一把抓住葉子往上拉，每一根都是驚喜。

看著孩子們帶著泥土味回教室，這大概就是農耕課最美的樣子 ❤️

請根據以上關鍵字與年級，直接產出課程側記（不需要詢問細節）："""


def build_tutoring_prompt(keywords: str, subject: str, grade: str, tone: str, gender: str = "男", name: str = "") -> str:
    tone_desc = TONES.get(tone, TONES["溫暖鼓勵"])
    noun, pronoun = _student_ref(gender, name)
    return f"""你是一位家教老師的文字助手，負責撰寫給學生的課後評語。

【年級】{grade}
【科目】{subject}
【語氣】{tone}（{tone_desc}）
【學生稱呼】請以「{noun}」稱呼這位學生，代名詞用「{pronoun}」
【本次關鍵字／重點】{keywords}

【格式要求】
- 使用繁體中文
- 字數 80–150 字
- 評語自然流暢，符合家教老師書寫課後評語的習慣
- 可以口語一點，像在跟學生說話
- 可用驚嘆號表現鼓勵！
- Emoji 可選用 1–2 個，不強制
- 直接輸出評語本文，不需要標題或項目符號

{FORBIDDEN_WORDS_NOTE}

請直接產出評語："""


def build_parent_report_prompt(keywords: str, subject: str, grade: str, tone: str, gender: str = "男", name: str = "") -> str:
    tone_desc = TONES.get(tone, TONES["溫暖鼓勵"])
    noun, pronoun = _student_ref(gender, name)
    return f"""你是一位家教老師的文字助手，負責撰寫給家長的學習側記（風格類似老師在家長群組的留言）。

【年級】{grade}
【科目】{subject}
【語氣】{tone}（{tone_desc}）
【學生稱呼】請以「{noun}」稱呼這位學生，代名詞用「{pronoun}」
【本次關鍵字／重點】{keywords}

【文章固定結構】
1. 標題：本週學習側記 ＋ 1 個 emoji（寫在最前面）
2. 首段（課程主題）：簡短說明今天／本週學習的主題與重點，約 40 字
3. 中段（學習過程）：描述學生的學習狀況、實際操作或解題過程，活潑有趣，約 80–100 字
4. 末段（溫馨結尾）：一句有溫度的收尾，讓家長感受到孩子的成長，約 20–30 字

全文總字數：150–200 字

【語氣規則】
- 口語自然，像老師在跟家長群組聊天的感覺
- 可用破折號——製造節奏感
- 可用驚嘆號表現學生的進步！
- Emoji 全文使用 2–4 個，分散在不同段落，不集中

{FORBIDDEN_WORDS_NOTE}

請直接產出學習側記（不需要詢問細節）："""


def build_school_comment_prompt(
    keywords: str, grade: str, tone: str, guides: list, gender: str = "男", name: str = ""
) -> str:
    tone_desc = TONES.get(tone, TONES["溫暖鼓勵"])
    guide_list = [SCHOOL_GUIDES[g] for g in guides if g in SCHOOL_GUIDES]
    guide_text = "、".join(guide_list) if guide_list else "整體學習表現"
    keywords_line = f"請自然融入以下關鍵字或概念：{keywords}\n" if keywords else ""
    noun, pronoun = _student_ref(gender, name)

    return f"""請為一位學生撰寫一段學校老師評語（學期末評語）。

【年級】{grade}
【語氣】{tone}（{tone_desc}）
【評語重點】{guide_text}
【學生稱呼】請以「{noun}」稱呼這位學生，代名詞用「{pronoun}」
{keywords_line}
【格式要求】
- 使用繁體中文
- 字數嚴格控制在 80–150 字之間
- 評語自然流暢，符合學校老師書寫評語的用語習慣（正式中帶溫度）
- 具體有感，避免空泛套語
- 不加任何標題或前言，直接輸出評語本文

{FORBIDDEN_WORDS_NOTE}

請直接輸出評語："""


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        grades=GRADES,
        tones=list(TONES.keys()),
        school_guides=list(SCHOOL_GUIDES.keys()),
    )


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    mode = data.get("mode", "farming")
    gender = data.get("gender", "男")
    name = data.get("name", "")

    if mode == "farming":
        prompt = build_farming_prompt(
            keywords=data.get("keywords", ""),
            grade=data.get("grade", "國一"),
            gender=gender,
            name=name,
        )
    elif mode == "tutoring":
        prompt = build_tutoring_prompt(
            keywords=data.get("keywords", ""),
            subject=data.get("subject", "數學"),
            grade=data.get("grade", "國一"),
            tone=data.get("tone", "溫暖鼓勵"),
            gender=gender,
            name=name,
        )
    elif mode == "parent_report":
        prompt = build_parent_report_prompt(
            keywords=data.get("keywords", ""),
            subject=data.get("subject", "數學"),
            grade=data.get("grade", "國一"),
            tone=data.get("tone", "溫暖鼓勵"),
            gender=gender,
            name=name,
        )
    elif mode == "school_comment":
        prompt = build_school_comment_prompt(
            keywords=data.get("keywords", ""),
            grade=data.get("grade", "國一"),
            tone=data.get("tone", "溫暖鼓勵"),
            guides=data.get("guides", []),
            gender=gender,
            name=name,
        )
    else:
        return jsonify({"error": "未知模式"}), 400

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    return jsonify({"feedback": text, "char_count": len(text)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
