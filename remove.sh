#!/bin/bash

# 삭제할 폴더 패턴을 정의합니다.
# 예시: 'test_'로 시작하는 모든 폴더를 찾습니다.
PATTERN="*sliced_on_1_0_1*"

# 삭제를 시작할 상위 디렉터리 경로를 정의합니다.
# 현재 디렉터리에서 삭제하려면 "."을 사용합니다.
SEARCH_PATH="/data/jhahn/data/shape_dataset/data/brain_lightsheet"

echo "경로: ${SEARCH_PATH} 에서 '${PATTERN}' 패턴을 가진 폴더를 삭제합니다."

# find 명령어를 사용하여 패턴에 맞는 디렉터리를 찾고 삭제합니다.
# -type d: 디렉터리만 찾습니다.
# -name "${PATTERN}": 지정된 패턴과 이름이 일치하는 디렉터리를 찾습니다.
# -exec rm -rf {} +: 찾아낸 디렉터리들을 rm -rf 명령으로 삭제합니다.
find "${SEARCH_PATH}" -type d -name "${PATTERN}" -exec rm -rdf {} +

echo "삭제가 완료되었습니다."
