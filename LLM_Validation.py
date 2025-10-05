from openai import OpenAI
from dotenv import load_dotenv
import os
import json



# 1) .env에서 OPENAI_API_KEY를 읽어옵니다.
load_dotenv()
IMG_BASE_URL =  os.getenv("IMG_BASE_URL")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_llm_with_raw_inputs(
    *,
    image_url: str,           # 이미지 URL
    analyze_id: int,          # id 값
    result_json_raw: str,     # json raw 값
    user_prompt: str = (
        # 지시 프롬포트
        "아래 이미지와 JSON 탐지 결과를 기준으로 파손 타당성과 위험도를 평가해줘."
    ),
    model: str = "gpt-5",     
) -> dict:
    """
    [함수의 역할]
    - 위의 4가지 입력(image_url, analyze_id, result_json_raw, user_prompt)을 LLM에 전달
    - LLM이 반드시 JSON으로만 답하게 강제
    - dict로 파싱해서 돌려줌

    [LLM에게 요구하는 출력 형태(스키마)]
    {
      "ok": boolean,                              # 전체 검증 통과 여부
      "verified_count": number,                   # ACCEPT 수
      "rejected_count": number,                   # REJECT 수
      "per_box": [                                # 박스별 판단
        {"index": number, "decision": "ACCEPT"|"REJECT", "reason": string}
      ],
      "risk_level": "LOW"|"MEDIUM"|"HIGH",        # 전체 위험도
      "comments": string                           # 간단 코멘트/권고
    }
    """
    # 시스템 가이드 (중요)
    guide = """
You are a road-damage quality inspector.
You will receive:
  (A) a road image (URL),
  (B) a JSON string that contains detections with bbox = [xmin, ymin, xmax, ymax] (pixel coordinates) and a Korean damage type label.

Your job:
  1) For each bbox, verify whether the label matches the visual evidence (ACCEPT/REJECT) and give a short reason.
  2) Compute risk scores strictly by the deterministic rules below (no alternative heuristics).
  3) Produce ONLY a JSON object that follows the schema at the end.

Deterministic risk rules (use these exactly):
- Frame size: Determine the image pixel width (W) and height (H). If exact size is unclear, infer from the image. As a fallback, if bbox coordinates appear to be pixel indices, you may use:
    W ≈ max(xmax across detections) + 1
    H ≈ max(ymax across detections) + 1
  Note: Prefer the actual image resolution if available.

- Area percent p(%) for each bbox i:
    p_i = 100 * ((xmax_i - xmin_i) * (ymax_i - ymin_i)) / (W * H)

- Type weights w_type and area thresholds p0(%) by Korean label:
    w_type:
      포트홀:1.00, 거북등:0.95, 러팅:0.90, 코루게이션및쇼빙:0.80,
      함몰:0.80, 박리:0.65, 단부균열:0.60, 밀림균열:0.60,
      세로방향균열:0.55, 반사균열:0.55, 시공균열:0.50, 라벨링:0.10
    p0:
      포트홀:1.5, 거북등:1.0, 러팅:2.0, 코루게이션및쇼빙:1.5,
      함몰:1.5, 박리:1.0, 단부균열:0.8, 밀림균열:0.8,
      세로방향균열:0.6, 반사균열:0.6, 시공균열:0.5, 라벨링:0.5

- Area effect g(p):
    g(p) = min(1, p / p0)

- Per-box risk score R_box (0..100):
    R_box = 100 * w_type * (0.5 + 0.5 * g(p))

- Frame-level aggregation:
    Let N be the number of detections (after verification decision is made, still compute R_box for each reported detection).
    R_max = max_i R_box_i
    R_avg = (sum_i R_box_i) / N
    count_boost = 8 * ln(1 + N)             # natural log

    R_frame = clamp( 0.6 * R_max + 0.4 * R_avg + count_boost , 0 , 100 )

- Risk level mapping:
    if R_frame < 33  -> "LOW"
    else if R_frame < 66 -> "MEDIUM"
    else               -> "HIGH"

Output schema (JSON only, no extra text):
{
  "ok": boolean,
  "verified_count": number,
  "rejected_count": number,
  "per_box": [
    {
      "index": number,                 # detection index in the input order (0-based)
      "label": string,                 # Korean damage label from input
      "bbox": [xmin, ymin, xmax, ymax],
      "area_percent": number,          # p_i computed
      "w_type": number,
      "p0": number,
      "g": number,                     # g(p_i)
      "r_box": number,                 # R_box_i
      "decision": "ACCEPT" | "REJECT",
      "reason": string                 # short justification
    }
  ],
  "r_max": number,
  "r_avg": number,
  "count_boost": number,
  "r_frame": number,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "comments": string                   # brief summary/maintenance suggestion
}
Return only the JSON object above, computed strictly by these rules.
"""

    
    
    # user 메시지에는 이미지 URL + 프롬프트 + 식별자 + result_json(raw) 전달
    messages = [
        {"role": "system", "content": guide},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": f"analyze_id: {analyze_id}"},
                {"type": "text", "text": f"result_json(raw): {result_json_raw}"},
            ],
        },
    ]

    # json으로 응답
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},  # json으로 응답
        )
        text = resp.choices[0].message.content or ""
    except TypeError:
        
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI API 호출 실패: {e}")
    except Exception as e:
        raise RuntimeError(f"OpenAI API 호출 실패: {e}")

    # 4) 응답 텍스트를 JSON으로 파싱
    try:
        return json.loads(text)
    except Exception:
        # JSON 형식 위반 시
        return {"_parse_error": True, "_raw": text}

def save_json(obj: dict, path: str) -> None:
    """dict를 보기 좋게 저장합니다."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 이미지 URL
    image_url = f"{IMG_BASE_URL}2025-09-13/7752_20250913.jpg"


    # analyze_id + result_json
    record = {
      "analyze_id": 47633,
      "result_json": "{\"cctv\": {\"id\": 7752, \"coordx\": 127.051682, \"coordy\": 36.198040, \"cctvname\": \"[천안논산선] 화정2교 ↑\"}, \"detections\": [{\"id\": 8070, \"bbox\": [343, 363, 373, 429], \"damage_type\": \"세로방향균열\"}, {\"id\": 8071, \"bbox\": [308, 286, 328, 323], \"damage_type\": \"세로방향균열\"}, {\"id\": 8072, \"bbox\": [436, 290, 475, 324], \"damage_type\": \"세로방향균열\"}, {\"id\": 8073, \"bbox\": [375, 256, 414, 268], \"damage_type\": \"반사균열\"}, {\"id\": 8074, \"bbox\": [376, 256, 415, 268], \"damage_type\": \"반사균열\"}, {\"id\": 8075, \"bbox\": [375, 256, 415, 268], \"damage_type\": \"반사균열\"}], \"analyzed_date\": \"2025-09-13\"}"
    }

    
    # 프롬프트
    user_prompt = (
        "해당 bbox가 라벨과 형태적으로 타당한지 박스별로 판정(ACCEPT/REJECT)하고, "
        "간단한 이유를 덧붙여줘. 전체 위험도(LOW/MEDIUM/HIGH)도 산정해줘. "
        "반드시 JSON만 반환해."
    )

    # 5) LLM 호출
    result = ask_llm_with_raw_inputs(
        image_url=image_url,
        analyze_id=record["analyze_id"],
        result_json_raw=record["result_json"],
        user_prompt=user_prompt,
        model="gpt-5",     
        # model="gpt-4o",
    )

    # 6) 결과 확인 + 저장
    print("=== LLM 결과(JSON) ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    out_path = f"verify_result_{record['analyze_id']}.json"
    save_json(result, out_path)
    print(f"\nSaved: {out_path}")