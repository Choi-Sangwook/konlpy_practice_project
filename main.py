# 정규표현식을 사용하기 위한 re 모듈을 불러옵니다.
import re

# PyTorch 텐서 처리를 위해 torch를 불러옵니다.
import torch

# 표 형태 데이터 처리를 위해 pandas를 불러옵니다.
import pandas as pd

# 그래프 출력을 위해 matplotlib을 불러옵니다.
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 워드 클라우드 생성을 위해 WordCloud를 불러옵니다.
from wordcloud import WordCloud

# 한국어 형태소 분석을 위해 Okt 형태소 분석기를 불러옵니다.
from konlpy.tag import Okt

#================================================================================
# 텍스트 수집
#================================================================================
# 기사 원본 링크 : https://m.sports.naver.com/fifaworldcup2026/article/076/0004418630

text = """
韓 실수 없이 조 1위 했더라면…"땡큐 코리아" 꽃길 예약한 멕시코, 32강 상대 스코틀랜드-카보베르데-사우디 중 한 팀 '유력'
[스포츠조선 윤진만 기자]'그 길은 우리가 걸었어야 하는 꽃길.'

대한민국 월드컵대표팀을 꺾고 조 1위로 32강에 오른 멕시코가 토너먼트에서 상대적으로 약한 팀들과 맞붙을 가능성이 커보인다.

스포츠 전문매체 '디애슬레틱'은 23일(한국시각), 멕시코가 32강전에서 맞붙을 팀으로 스코틀랜드, 카보베르데, 사우디아라비아 등을 거론했다. 2026년 북중미월드컵 A조 1위를 차지한 멕시코는 대진상 C, E, F, H, I조 3위 중 한 팀과 32강에서 격돌한다.

23일 현재, C조 3위가 스코틀랜드(1승1패)다. '디애슬레틱'은 멕시코가 스코틀랜드와 맞붙을 확률을 가장 높은 38%로 예측했다. 다음으로 높은 게 '돌풍팀'인 조 3위 카보베르데(2무)다. 이번에 월드컵에 처음 나선 카보베르데는 스페인(0대0 무), 우루과이(2대0 무)와 비기는 저력을 선보였다.
사우디는 H조 4위에 처져있다. 우루과이와 1대1로 비기고 스페인에 0대4로 패해 1무1패 승점 1점을 기록했다. '디애슬레틱'은 사우디가 멕시코의 상대가 될 확률을 12%로 예측했다.

FIFA 랭킹 14위인 멕시코의 예상 상대 세 팀의 랭킹은 스코틀랜드 42위, 카보베르데 67위, 사우디 61위다. 조 1위만이 만끽할 수 있는 여유다. 게다가 멕시코는 32강전을 7월 1일 '멕시코 축구의 성지'인 멕시코시티의 2200m 고지대 경기장인 아스테카 스타디움에서 치른다.

32강에서 승리하면 16강전도 이동없이 아스테카 스타디움에서 격돌한다. A조 1위는 결국 멕시코를 위한 '꽃길' 대진이었던 셈이다.

반면 한국이 가시밭길이 에고돼있다. 조별리그 1차전 체코전 승리로 조 1위까지 넘봤지만, 2차전에서 멕시코에 0대1로 패하며 조 1위 확보에 실패해써다.

1승1패 승점 3점을 기록하며 조 2위를 달리는 한국은 25일 남아공(승점 1)과의 경기에서 비기기만 해도 체코(승점 1)와 남아공을 따돌리고 2위를 확정할 수 있다.

한국이 조 2위를 유지하면 29일 LA에서 B조 2위인 스위스(승점 4)과 8강 진출권을 노린다.
"""
#================================================================================
#텍스트 정제
#================================================================================
cleaned_text = re.sub(r"[^0-9ㄱ-ㅎㅏ-ㅣ가-힣\s]", " ", text)
cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
print('정제된 텍스트')
print(cleaned_text)

#================================================================================
# 형태소 분석
#================================================================================
okt = Okt()
print(okt.phrases(cleaned_text))
print(okt.morphs(cleaned_text))
print(okt.nouns(cleaned_text))
print(okt.pos(cleaned_text))

nouns = okt.nouns(cleaned_text)

# Okt가 숫자+한글 조합을 명사로 잘 잡지 못할 수 있어서
# 분석에 남기고 싶은 토너먼트 단계 표현은 따로 보존합니다.
preserved_words = re.findall(r"\d+\s*강", cleaned_text)
preserved_words = [word.replace(" ", "") for word in preserved_words]
nouns.extend(preserved_words)
#================================================================================
# 불용어 제거
#================================================================================
stopwords = [
    "스포츠조선", "윤진만", "기자",
    "디애", "슬레", "디애슬레틱",
    "전문", "매체", "한국시각",
    "예측", "기록", "다음",
    "처음", "결국", "반면", "셈"
]
nouns = [
    word
    for word in nouns
    if len(word) > 1 and word not in stopwords
]
print("\n추출된 명사:")
print(nouns)

#================================================================================
# PyTorch Tensor 변환
#================================================================================
# 중복을 제거한 단어 목록을 정렬하여 vocabulary를 만듭니다.
vocab = sorted(set(nouns))

# 단어를 숫자 인덱스로 바꾸기 위한 딕셔너리를 만듭니다.
word_to_id = {word: idx for idx, word in enumerate(vocab)}

# 각 명사를 숫자 ID로 변환합니다.
word_ids = [word_to_id[word] for word in nouns]

# 숫자 ID 리스트를 PyTorch 텐서로 변환합니다.
word_counts_tensor = torch.tensor(word_ids)


#================================================================================
# 단어 빈도 계산
#================================================================================
# torch.bincount()는 각 숫자 ID가 몇 번 나왔는지 계산합니다.
word_counts = torch.bincount(word_counts_tensor)

# PyTorch 텐서를 파이썬 딕셔너리 형태로 변환합니다.
word_freq = {
    vocab[i]: int(word_counts[i].item())
    for i in range(len(vocab))
}

# 빈도수가 높은 순서대로 정렬합니다.
word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

# 빈도 결과를 출력합니다.
print("\n단어 빈도:")
print(word_freq)


#================================================================================
# 상위 20개 단어 시각화
#================================================================================

# 단어 빈도 딕셔너리를 pandas Series로 변환합니다.
word_freq_series = pd.Series(word_freq)

# 상위 20개 단어를 출력합니다.
top_words = word_freq_series.head(20)
print("\n상위 단어:")
print(top_words)


#================================================================================
# 워드클라우드와 상위 단어 막대그래프 생성
#================================================================================
font_path = 'fonts/malgunsl.ttf'
font_prop = font_manager.FontProperties(fname=font_path)

# WordCloud 객체를 생성합니다.
wordcloud = WordCloud(
    font_path=font_path,
    background_color="white",
    width=1000,
    height=800,
).generate_from_frequencies(word_freq)

# 워드클라우드와 상위 20개 단어 빈도를 함께 보여줍니다.
fig, (ax_cloud, ax_bar) = plt.subplots(1, 2, figsize=(18, 8))

ax_cloud.imshow(wordcloud)
ax_cloud.axis("off")
ax_cloud.set_title("워드클라우드", fontproperties=font_prop, fontsize=16)

bar_words = top_words.sort_values()
y_positions = range(len(bar_words))
ax_bar.barh(y_positions, bar_words.values, color="#4C78A8")
ax_bar.set_yticks(y_positions)
ax_bar.set_yticklabels(bar_words.index, fontproperties=font_prop)
ax_bar.set_xlabel("등장 횟수", fontproperties=font_prop)
ax_bar.set_title("상위 20개 단어 빈도", fontproperties=font_prop, fontsize=16)

for index, value in enumerate(bar_words.values):
    ax_bar.text(value + 0.05, index, str(value), va="center", fontproperties=font_prop)

ax_bar.set_xlim(0, max(bar_words.values) + 1)
fig.suptitle("단어 빈도 분석 결과", fontproperties=font_prop, fontsize=20)
fig.tight_layout()

# 결과 이미지를 파일로 저장합니다.
output_path = "wordcloud_top20.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n이미지 저장 완료: {output_path}")

# 그래프를 출력합니다.
plt.show()
