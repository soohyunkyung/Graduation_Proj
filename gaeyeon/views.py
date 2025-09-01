import os
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
from graduation_proj import settings

# 모델 로딩 (앱 시작 시 1회만)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'gaeyeon', 'dog_breed_classifier.h5')
model = load_model(MODEL_PATH)

# 클래스 이름들 (예: ['beagle', 'chihuahua', 'poodle', ...])
CLASS_NAMES = [
    'bostonbull', 'dingo', 'pekinese', 'bluetick', 'goldenretriever',
    'bedlingtonterrier', 'borzoi', 'basenji', 'scottishdeerhound',
    'shetlandsheepdog', 'walkerhound', 'maltesedog', 'norfolkterrier',
    'africanhuntingdog', 'wire-hairedfoxterrier', 'redbone', 'lakelandterrier',
    'boxer', 'doberman', 'otterhound', 'standardschnauzer', 'irishwaterspaniel',
    'black-and-tancoonhound', 'cairn', 'affenpinscher', 'labradorretriever',
    'ibizanhound', 'englishsetter', 'weimaraner', 'giantschnauzer', 'groenendael',
    'dhole', 'toypoodle', 'borderterrier', 'tibetanterrier', 'norwegianelkhound',
    'shih-tzu', 'irishterrier', 'kuvasz', 'germanshepherd',
    'greaterswissmountaindog', 'basset', 'australianterrier', 'schipperke',
    'rhodesianridgeback', 'irishsetter', 'appenzeller', 'bloodhound', 'samoyed',
    'miniatureschnauzer', 'brittanyspaniel', 'kelpie', 'papillon', 'bordercollie',
    'entlebucher', 'collie', 'malamute', 'welshspringerspaniel', 'chihuahua',
    'saluki', 'pug', 'malinois', 'komondor', 'airedale', 'leonberg',
    'mexicanhairless', 'bullmastiff', 'bernesemountaindog',
    'americanstaffordshireterrier', 'lhasa', 'cardigan', 'italiangreyhound',
    'clumber', 'scotchterrier', 'afghanhound', 'oldenglishsheepdog',
    'saintbernard', 'miniaturepinscher', 'eskimodog', 'irishwolfhound',
    'brabancongriffon', 'toyterrier', 'chow', 'flat-coatedretriever',
    'norwichterrier', 'soft-coatedwheatenterrier', 'staffordshirebullterrier',
    'englishfoxhound', 'gordonsetter', 'siberianhusky', 'newfoundland', 'briard',
    'chesapeakebayretriever', 'dandiedinmont', 'greatpyrenees', 'beagle',
    'vizsla', 'westhighlandwhiteterrier', 'kerryblueterrier', 'whippet',
    'sealyhamterrier', 'standardpoodle', 'keeshond', 'japanesespaniel',
    'miniaturepoodle', 'pomeranian', 'curly-coatedretriever', 'yorkshireterrier',
    'pembroke', 'greatdane', 'blenheimspaniel', 'silkyterrier', 'sussexspaniel',
    'germanshort-hairedpointer', 'frenchbulldog', 'bouvierdesflandres',
    'tibetanmastiff', 'englishspringer', 'cockerspaniel', 'rottweiler'
]


def predict_breed(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    print(pred)
    return CLASS_NAMES[np.argmax(pred)]

@csrf_exempt
def upload_view(request):
    if request.method == 'POST':
        print(request)
        print(request.FILES)
        if 'fileInput' in request.FILES:
            file = request.FILES['fileInput']
            file_path = default_storage.save('uploads/' + file.name, file)
            full_path = os.path.join(default_storage.location, file_path)
            print(full_path)

            breed = predict_breed(full_path)
            print(breed)
            request.session['breed'] = breed
            return JsonResponse({'success': True, 'redirect': '/survey/'})

        elif 'breed_text' in request.POST:
            input_breed = request.POST['breed_text']
            # 전처리: 소문자 + 공백/스코어 제거
            cleaned = input_breed.lower().replace(" ", "").replace("-", "").replace("_", "")
            request.session['breed'] = cleaned
            return JsonResponse({'success': True, 'redirect': '/survey/'})
        return JsonResponse({'success': False, 'error': '입력 누락: image나 breed_text가 필요합니다.'})


    return render(request, 'upload.html')


csv_path = os.path.join(settings.BASE_DIR, 'gaeyeon', 'dog_breeds_info.csv')
df = pd.read_csv(csv_path)

def survey_view(request):
    breed = request.session.get('breed', '견종 정보 없음')
    if request.method == 'POST':
        # 질문 응답 수집
        q = {f'q{i}': request.POST.get(f'q{i}') for i in [11,12,13,14,15,16,17,19,20,21,23,24,25,26,27,28,31,32,33,34,35,37,38,39,40,43,44,45]}

        # 견종 데이터 가져오기
        breed_info = df[df['Breed'] == str(breed)].iloc[0]
        height = breed_info['Height_avg']
        longevity = breed_info['Longevity_avg']
        fur_color = breed_info['Fur Color']
        eye_color = breed_info['Color of Eyes']
        traits = breed_info['Character Traits']

        # 실패 플래그 정의
        economic_weak = q['q11'] == '아니요' or q['q13'] in ['5만원 미만', '5만~ 10만원 미만'] or q['q14'] in ['5만원 미만', '감당하기 어렵다'] or q['q19'] == '감당 어렵다'
        environment_weak = q['q20'] == '아니요' or q['q23'] == '불가능하다' or q['q25'] == '없다'
        irresponsible = q['q26'] in ['거의 함께할 수 없다', '1시간 내외'] or q['q27'] in ['거의 함께할 수 없다', '1시간 내외'] or q['q28'] == '거의 하지 않을 것 같다' or q['q32'] == '어려울 것 같다' or q['q34'] == '바로 쉬어야 함'

        # 견종 크기 불일치
        size_map = {
            'miniminidog': height > 11,
            'minidog': not (12 <= height <= 15),
            'middledog': not (16 <= height <= 21),
            'bigdog': not (22 <= height <= 27),
            'bigbigdog': height < 28
        }
        fail_q38 = size_map.get(q['q38'], False)

        # 수명 불일치
        if q['q40'] == 'under10':
            fail_q40 = longevity >= 10
        elif q['q40'] == '10~13':
            fail_q40 = not (10 <= longevity <= 13)
        elif q['q40'] == 'over13':
            fail_q40 = longevity <= 13
        else:
            fail_q40 = False

        # 색상 및 특성 불일치
        fail_q39 = q['q39'] not in eye_color
        fail_q43 = q['q43'] != 'all_ok' and q['q43'] not in fur_color
        fail_q44 = q['q44'] not in eye_color
        fail_traits = not any(t in traits for t in q['q45'].split(','))
        fail_fur_match = not any(t in fur_color for t in q['q37'].split(','))
        trait_unmatch = fail_q38 or fail_q39 or fail_q40 or fail_traits or fail_q43 or fail_q44 or fail_fur_match

        # 복합 실패 조건 우선 적용
        if economic_weak and environment_weak and irresponsible and trait_unmatch:
            return render(request, 'fail.html', {'fail_reason': '경제적 미비, 환경적 미비, 책임감 부족, 견종과 성향 불일치'})
        if economic_weak and environment_weak and irresponsible:
            return render(request, 'fail.html', {'fail_reason': '경제적 미비, 환경적 미비, 책임감 부족'})
        if economic_weak and environment_weak and trait_unmatch:
            return render(request, 'fail.html', {'fail_reason': '경제적 미비, 환경적 미비, 견종과 성향 불일치'})
        if economic_weak and irresponsible and trait_unmatch:
            return render(request, 'fail.html', {'fail_reason': '경제적 미비, 책임감 부족, 견종과 성향 불일치'})
        if environment_weak and irresponsible and trait_unmatch:
            return render(request, 'fail.html', {'fail_reason': '환경적 미비, 책임감 부족, 견종과 성향 불일치'})
        if economic_weak and environment_weak:
            return render(request, 'fail.html', {'fail_reason': '경제적 미비 및 환경적 미비'})
        if economic_weak and irresponsible:
            return render(request, 'fail.html', {'fail_reason': '경제적 미비 및 책임감 부족'})
        if economic_weak and trait_unmatch:
            return render(request, 'fail.html', {'fail_reason': '경제적 미비 및 견종과 성향 불일치'})
        if environment_weak and irresponsible:
            return render(request, 'fail.html', {'fail_reason': '환경적 미비 및 책임감 부족'})
        if environment_weak and trait_unmatch:
            return render(request, 'fail.html', {'fail_reason': '환경적 미비 및 견종과 성향 불일치'})
        if irresponsible and trait_unmatch:
            return render(request, 'fail.html', {'fail_reason': '책임감 부족 및 견종과 성향 불일치'})

        # 단일 실패 조건
        if economic_weak:
            return render(request, 'fail.html', {'fail_reason': '경제적 미비'})
        if environment_weak:
            return render(request, 'fail.html', {'fail_reason': '환경적 미비'})
        if irresponsible:
            return render(request, 'fail.html', {'fail_reason': '책임감 부족'})
        if trait_unmatch:
            return render(request, 'fail.html', {'fail_reason': '견종과 성향 불일치'})

        # 논리 오류 1회 경고
        warning_conditions = [
            q['q12'] == '200만원 미만' and q['q13'] == '20만원 이상',
            q['q13'] == '20만원 이상' and q['q16'] in ['5만원 미만', '5만 ~ 10만원 미만'],
            q['q11'] == '예' and q['q14'] == '5만원 미만',
            q['q15'] == '대출 또는 외부 도움을 받았다' and q['q19'] == '감당 어렵다',
            q['q13'] in ['5만원 미만', '5만 ~ 10만원 미만', '10만 ~ 20만원 미만'] and q['q14'] == '50만 원 이상',
            q['q20'] == '예' and q['q23'] in ['불가능하다', '잘 모르겠다'],
            q['q20'] == '예' and q['q25'] == '없다',
            q['q20'] == '예' and q['q24'] == '1년 내 계획이 있다',
            q['q17'] == '아직 생각 안함' and q['q21'] == '6개월 이상',
            q['q26'] in ['거의 함께할 수 없다', '1시간 내외'] and q['q28'] in ['거의 하지 않을 것 같다', '주 1~4회'],
            q['q27'] in ['거의 함께할 수 없다', '1시간 내외'] and q['q28'] == '거의 매일 산책할 수 있다',
            q['q26'] == '거의 함께할 수 없다' and q['q31'] == '5시간 이상',
            q['q28'] in ['주 1~4회', '거의 매일 산책할 수 있다'] and q['q34'] == '바로 쉬어야 함',
            q['q28'] in ['주 1~4회', '거의 매일 산책할 수 있다'] and q['q35'] in ['유동적이어서 계획이 자주 바뀐다', '매일 다르고 예측하기 어렵다'],
            q['q32'] == '어려울 것 같다' and q['q33'] in ['놀이/훈련 상호작용', '건강관리 철저']
        ]
        if any(warning_conditions):
            if request.session.get('survey_warning_once'):
                return render(request, 'fail.html', {'fail_reason': '설문 문항 내 논리적 오류'})
            else:
                request.session['survey_warning_once'] = True
                return render(request, 'survey.html', {'warning': '설문을 진실 되게 작성 하세요.'})

        return render(request, 'pass.html')

    return render(request, 'survey.html', {'breed': breed})
