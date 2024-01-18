import requests
import urllib
import n_api

def shop_search(client_Id, client_Secret, query, maker, max_price):
    try:
        url = 'https://openapi.naver.com/v1/search/shop.json'

        headers = {
            'X-Naver-Client-Id': client_Id,
            'X-Naver-Client-Secret': client_Secret
        }

        items = []
        start = 1

        while True:

            params = {
                'query': query,
                'display': 100,
                'maker': maker,
                'start': start
            }

            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                result_json = response.json()
                current_items = result_json.get('items', [])

                if not current_items:
                    break

                items.extend(current_items)
                start += len(current_items)

            else:
                print(f'Response Error Code: {response.status_code}')
                return 0
            
            filtered_items = [
                
                item for item in items if maker in item.get('maker', '') and int(item.get('lprice', 0)) <= max_price
            ]

            return filtered_items
    
    except Exception as e:
        print(f'Exception: {e}')
        return 0


if __name__ == '__main__':
    client_Id = n_api.client_Id
    client_Secret = n_api.client_Secret
    query = '노트북'
    maker = '삼성전자'
    max_price = 1000000

    result = shop_search(client_Id, client_Secret, query, maker, max_price)

    if result:
        for idx, item in enumerate(result, 1):
            print(f'{idx}, 상품명: {item.get('title')}, 제조사: {item.get('maker')}, 가격: {item.get('lprice')}원')
    else:
        print('검색 결과가 없습니다.')
    # print(result)
