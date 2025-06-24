import os

# 삭제 대상 폴더 경로
target_dir = "./Data/byte_16/postgres"  # ← 여기에 실제 폴더 경로 입력

# 삭제 기준 번호
start_number = 20001

# 삭제한 파일 개수 카운터
deleted = 0

for filename in os.listdir(target_dir):
    if filename.startswith("packet_") and filename.endswith(".png"):
        try:
            # 숫자 부분만 추출
            number = int(filename.replace("packet_", "").replace(".png", ""))
            if number >= start_number:
                file_path = os.path.join(target_dir, filename)
                os.remove(file_path)
                deleted += 1
        except ValueError:
            # 숫자로 파싱 실패한 파일은 무시
            continue

print(f"✅ {deleted}개 파일 삭제 완료 (packet_{start_number}.png 이상)")
