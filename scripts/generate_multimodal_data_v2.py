import os
import json
import random
from datetime import datetime, timedelta
import pandas as pd

DATA_DIR = "research/data/processed/synthetic_personas/"
OUT_DIR = "research/data/processed/synthetic_personas_multimodal/"
SELL_PRICES = "research/m5-forecasting-accuracy/sell_prices.csv"
CALENDAR = "research/m5-forecasting-accuracy/calendar.csv"
os.makedirs(OUT_DIR, exist_ok=True)

SELECTED_FILES = ["CA_1_P1.json", "CA_1_P2.json"]

def get_high_price_items(store_id, df, quantile=0.75):
    store_items = df[df['store_id']==store_id]
    threshold = store_items['sell_price'].quantile(quantile)
    high_items = store_items[store_items['sell_price']>=threshold]
    result = (high_items.groupby('item_id')['sell_price'].mean()).reset_index()
    return result

def sample_premium_receipt(store_id, item_pool, spend_range):
    trials = 0
    while trials < 100:
        num_items = random.randint(1, 3)
        items = item_pool.sample(num_items)
        receipt_items = []
        total = 0
        for _, row in items.iterrows():
            qty = random.randint(1,2)
            receipt_items.append({
                "item_id": row['item_id'],
                "price": round(float(row['sell_price']),2),
                "qty": qty
            })
            total += row['sell_price']*qty
        total = round(total,2)
        if spend_range[0] <= total <= spend_range[1]:
            return receipt_items, total
        trials += 1
    return None, None

def generate_dates(start_date, n, visit_freq_mean):
    interval = max(1, int(7/visit_freq_mean))
    dates = [start_date + timedelta(days=i*interval) for i in range(n)]
    return [d.strftime('%Y-%m-%d') for d in dates]

def regenerate_receipts_for_p2(profile, item_pool, spend_range):
    receipts = []
    i = 0
    today = datetime(2016,2,28) # 고정 기준일
    dates = generate_dates(today - timedelta(days=20), 20, profile['visit_freq_mean'])
    while len(receipts)<20:
        items, total = sample_premium_receipt("CA_1", item_pool, spend_range)
        if items is None: continue
        receipt = {
            "date": dates[i],
            "items": items,
            "total_spent": total
        }
        # 검증
        if round(sum(it['price']*it['qty'] for it in items),2) == round(total,2):
            receipts.append(receipt)
            i += 1
    return receipts

def regenerate_receipts_for_p1(profile, orig_history):
    today = datetime(2016,2,28)
    n = 20
    dates = generate_dates(today-timedelta(days=20), n, profile['visit_freq_mean'])
    out = []
    for i, h in enumerate(orig_history[:n]):
        h_new = h.copy()
        h_new['date'] = dates[i]
        # qty 병합 & 무결성 재확인
        items_dict = {}
        for it in h['items']:
            k = it['item_id']
            if k in items_dict:
                items_dict[k]['qty'] += 1
            else:
                items_dict[k] = {'item_id':k, 'price':it['price'], 'qty':1}
        h_new['items'] = list(items_dict.values())
        assert round(sum(x['price']*x['qty'] for x in h_new['items']),2) == round(h['total_spent'],2)
        out.append(h_new)
    return out

def main():
    prices = pd.read_csv(SELL_PRICES)
    with open(os.path.join(DATA_DIR, "CA_1_P1.json"),'r',encoding='utf-8') as f:
        data_p1 = json.load(f)
    with open(os.path.join(DATA_DIR, "CA_1_P2.json"),'r',encoding='utf-8') as f:
        data_p2 = json.load(f)
    # P2: 프리미엄 고가 샘플링/qty 제한/재생성
    high_pool = get_high_price_items("CA_1", prices)
    receipts_p2 = regenerate_receipts_for_p2(data_p2['profile'], high_pool, data_p2['profile']['spend_range'])
    # P1: realistic 날짜 생성, 병합 검증
    receipts_p1 = regenerate_receipts_for_p1(data_p1['profile'], data_p1['history'])
    out_p1 = {
        "persona_id": data_p1["persona_id"],
        "store_id": data_p1.get("store_id","CA_1"),
        "profile": data_p1["profile"],
        "history": receipts_p1
    }
    out_p2 = {
        "persona_id": data_p2["persona_id"],
        "store_id": data_p2.get("store_id","CA_1"),
        "profile": data_p2["profile"],
        "history": receipts_p2
    }
    with open(os.path.join(OUT_DIR,"CA_1_P1_FIXED.json"),'w',encoding='utf-8') as f:
        json.dump(out_p1,f,ensure_ascii=False,indent=2)
    with open(os.path.join(OUT_DIR,"CA_1_P2_FIXED.json"),'w',encoding='utf-8') as f:
        json.dump(out_p2,f,ensure_ascii=False,indent=2)
    print("파일 저장 완료:")
    for fname in ["CA_1_P1_FIXED.json", "CA_1_P2_FIXED.json"]:
        path = os.path.join(OUT_DIR, fname)
        print(f"- {path}, size={os.path.getsize(path)} bytes")
    # CA_1_P2 샘플 일부 출력
    with open(os.path.join(OUT_DIR,"CA_1_P2_FIXED.json"),'r',encoding='utf-8') as f:
        p2 = json.load(f)
    print("\n샘플: CA_1_P2_FIXED\n", json.dumps(p2,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
