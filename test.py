# 정규표현식을 사용하기 위한 re 모듈을 불러옵니다.
import re

# PyTorch 텐서 처리를 위해 torch를 불러옵니다.
import torch

# 표 형태 데이터 처리를 위해 pandas를 불러옵니다.
import pandas as pd

# 그래프 출력을 위해 matplotlib을 불러옵니다.
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

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
# 공통 함수
#================================================================================
FONT_PATH = "font/malgunsl.ttf"
font_prop = FontProperties(fname=FONT_PATH)


def clean_text(raw_text):
    cleaned = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\s]", " ", raw_text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def make_word_freq(words):
    words = [word for word in words if len(word.replace(" ", "")) > 1]

    if not words:
        return {}

    vocab = sorted(set(words))
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    word_ids = [word_to_id[word] for word in words]
    word_counts_tensor = torch.tensor(word_ids)
    word_counts = torch.bincount(word_counts_tensor)

    word_freq = {
        vocab[i]: int(word_counts[i].item())
        for i in range(len(vocab))
    }
    return dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))


def shorten_label(label, max_len=20):
    if len(label) <= max_len:
        return label
    return label[:max_len] + "..."


def draw_result(ax_cloud, ax_bar, title, word_freq):
    wordcloud = WordCloud(
        font_path=FONT_PATH,
        background_color="white",
        width=900,
        height=500,
    ).generate_from_frequencies(word_freq)

    ax_cloud.imshow(wordcloud)
    ax_cloud.set_title(f"{title} WordCloud", fontproperties=font_prop, fontsize=14)
    ax_cloud.axis("off")

    top_words = pd.Series(word_freq).head(20).sort_values()
    labels = [shorten_label(label) for label in top_words.index]

    ax_bar.barh(labels, top_words.values)
    ax_bar.set_title(f"{title} Top 20", fontproperties=font_prop, fontsize=14)
    ax_bar.tick_params(axis="x", labelsize=9)
    ax_bar.tick_params(axis="y", labelsize=8)

    for tick in ax_bar.get_yticklabels():
        tick.set_fontproperties(font_prop)


#================================================================================
# 형태소 분석
#================================================================================
cleaned_text = clean_text(text)
okt = Okt()

analysis_words = {
    "Nouns": okt.nouns(cleaned_text),
    "Phrases": okt.phrases(cleaned_text),
    "Morphs": okt.morphs(cleaned_text),
    "POS": [
        f"{word}/{tag}"
        for word, tag in okt.pos(cleaned_text)
        if len(word.replace(" ", "")) > 1
    ],
}


#================================================================================
# 빈도 계산 및 출력
#================================================================================
analysis_freqs = {
    name: make_word_freq(words)
    for name, words in analysis_words.items()
}

for name, word_freq in analysis_freqs.items():
    print(f"\n{name} 상위 20개:")
    print(pd.Series(word_freq).head(20))


#================================================================================
# 워드클라우드 및 상위 단어 시각화
#================================================================================
fig, axes = plt.subplots(
    nrows=4,
    ncols=2,
    figsize=(20, 24),
    gridspec_kw={"width_ratios": [1.3, 1]},
)

for row_index, (name, word_freq) in enumerate(analysis_freqs.items()):
    draw_result(
        axes[row_index][0],
        axes[row_index][1],
        name,
        word_freq,
    )

fig.suptitle("Okt 분석 방식별 단어 빈도 비교", fontproperties=font_prop, fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("test_okt_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
