# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def korean_to_be_englished(korean_word):
    r_lst = []
    for w in list(korean_word.strip()):
        ## 영어인 경우 구분해서 작성함. 
        if '가'<=w<='힣':
            ## 588개 마다 초성이 바뀜. 
            ch1 = (ord(w) - ord('가'))//588
            ## 중성은 총 28가지 종류
            ch2 = ((ord(w) - ord('가')) - (588*ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588*ch1) - 28*ch2
            r_lst.append([CHOSUNG_LIST[ch1], JUNGSUNG_LIST[ch2], JONGSUNG_LIST[ch3]])
        else:
            r_lst.append([w])
    return r_lst


# 입력한 문장의 자음, 모음 벡터화
def make_input_vector(text):
    # 자음, 모음 카데고리 사전 만들기
    NUMBER_LIST = list(map(str, [i for i in range(10)]))
    ALPHABET_LIST = [chr(i) for i in range(ord('a'), ord('z')+1)]
    SPECIAL_WORD_LIST = ['!', '"', '#', '$', '%', '&', '\'', '?','@', '*', '+',',','-','.', '/', '~', ':', '^', ' ']
    ONEWORD_LIST = ['ㄳ', 'ㄵ', 'ㅄ', 'ㄺ']

    # 초성, 중성, 종성 리스트 생성
    word_vector_list = CHOSUNG_LIST+JUNGSUNG_LIST+JONGSUNG_LIST+NUMBER_LIST+ALPHABET_LIST+SPECIAL_WORD_LIST+ONEWORD_LIST

    # 카데고리 사전의 각 데이터와 인덱스를 딕셔너리로 추출
    # 자음일 때 예를 들어 'ㄱ': [0, 41]은 ㄱ이 초성으로 오면 0, 종성으로 오면 41
    # ' '(공백) 이면 ' ': [40, 122]. 40은 글자에 종성이 없을 때. 122는 단어와 단어 사이 띄어쓰기

    word_vector_idx_dict = {}

    for idx, data in enumerate(word_vector_list):
        if not data in word_vector_idx_dict:
            word_vector_idx_dict[data] = [] 
        word_vector_idx_dict[data].append(idx)

    # 자음 중 초성과 종성에서 종성에만 있는 데이터를 따로 추출
    JONGSUNG_ONLY = sorted(list(set(JONGSUNG_LIST)-set(CHOSUNG_LIST)))[1:]
    
    input_vector = [[0 for _ in range(101)] for _ in range(200)]
    text = text.lower()
    consonants_and_vowels = korean_to_be_englished(text)

    if len(sum(consonants_and_vowels,[]))<200:
        cnt = 0  # input_vector의 index
        for words in consonants_and_vowels:
            if len(words) == 3:  # 한글일 때 (초성, 중성, 종성)
                for idx in range(3):
                    if idx == 0 or idx == 1 or (idx == 2 and (words[idx] == ' ' or words[idx] in JONGSUNG_ONLY)):
                        """
                        1. 초성
                        2. 중성 
                        3. 종성이 없을 때 
                        4. 종성이 있을 때, 자음 중 종성만 있으면
                        """
                        input_vector[cnt][word_vector_idx_dict[words[idx]][0]]=1
                    else:  # 자음 중 초성과 종성 둘 다 있는 경우, 종성
                        input_vector[cnt][word_vector_idx_dict[words[idx]][1]]=1
                    cnt += 1
                    
            else:  # 한글이 아닌 경우 (특수문자, 영어)
                if words[0] not in word_vector_list:  # word_vector_list에 없는 특수문자는 제외
                    pass
                elif words[0] == ' ':  # 띄어쓰기인 경우
                    input_vector[cnt][word_vector_idx_dict[words[0]][1]]=1
                else:  # 특수문자, 영어인 경우
                    input_vector[cnt][word_vector_idx_dict[words[0]][0]]=1
                cnt += 1
        
        return input_vector
    
    else:
        return print('Invalid Input')
    
if __name__ == "__main__":
    text = "취뽀하고 싶다. 나 좀 데려가줘~~"
    print(make_input_vector(text))