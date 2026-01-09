import cv2
import time
from ultralytics import YOLO
from pymcprotocol import Type3E

PLC_IP = "192.168.3.120"
PLC_PORT = 5010

SEND_INTERVAL = 0.2   # PLC 송신 주기 (초)

# ================= PLC 연결 =================
def connect_plc():
    plc = Type3E()
    plc.connect(PLC_IP, PLC_PORT)
    return plc

# ================= STRING 쓰기 =================
def write_string(plc, headdevice, text, max_words=10):
    ascii_codes = [ord(c) for c in text]
    words = []

    for i in range(0, len(ascii_codes), 2):
        if i + 1 < len(ascii_codes):
            word = ascii_codes[i] | (ascii_codes[i + 1] << 8)
        else:
            word = ascii_codes[i]
        words.append(word)

    while len(words) < max_words:
        words.append(0)

    plc.batchwrite_wordunits(headdevice, words)

# ================= INT 쓰기 =================
def write_int(plc, headdevice, value):
    plc.batchwrite_wordunits(headdevice, [int(value)])

# ================= PLC 데이터 송신 =================
def send_result_to_plc(plc, detected, class_name="", confidence=0):
    write_string(plc, "D1020", "START")

    if detected:
        write_string(plc, "D1040", class_name)
        write_int(plc, "D1050", confidence)
        write_string(plc, "D1060", "YES")
    else:
        write_string(plc, "D1040", "DEF")
        write_int(plc, "D1050", 0)
        write_string(plc, "D1060", "NO")

    write_string(plc, "D1030", "DONE")

# ================= 안전 송신 래퍼 =================
def safe_send_to_plc(plc, detected, class_name, confidence):
    try:
        send_result_to_plc(plc, detected, class_name, confidence)
        return True
    except Exception as e:
        print("[PLC ERROR]", e)
        return False

# ================= PLC 재연결 =================
def reconnect_plc():
    try:
        print("[PLC] 재연결 시도...")
        plc = connect_plc()
        print("[PLC] 재연결 성공")
        return plc
    except Exception as e:
        print("[PLC] 재연결 실패:", e)
        return None

# ================= 메인 루프 =================
def webcam_yolo_plc():
    model = YOLO("runs/detect/train6/weights/best.pt")

    plc = None
    plc_connected = False
    last_send_time = 0

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, conf=0.5)
        r = results[0]

        detected = False
        cls_name = "DEF"
        confidence = 0

        if r.boxes is not None and len(r.boxes) > 0:
            box = max(r.boxes, key=lambda b: float(b.conf[0]))
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            confidence = int(float(box.conf[0]) * 100)
            detected = True

        # -------- PLC 송신 주기 제한 --------
        now = time.time()
        if now - last_send_time >= SEND_INTERVAL:
            last_send_time = now

            # PLC 미연결 시 재연결
            if not plc_connected:
                plc = reconnect_plc()
                plc_connected = plc is not None

            # PLC 연결되어 있을 때만 송신
            if plc_connected:
                success = safe_send_to_plc(
                    plc,
                    detected,
                    cls_name,
                    confidence
                )

                # 송신 실패 → PLC 재연결 대기 상태
                if not success:
                    try:
                        plc.close()
                    except:
                        pass
                    plc = None
                    plc_connected = False

        annotated = r.plot()
        cv2.imshow("YOLO → PLC", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 = ESC
            print("프로그램 종료")
            break

    cap.release()
    cv2.destroyAllWindows()

    if plc_connected:
        plc.close()

# ================= 실행 =================
webcam_yolo_plc()