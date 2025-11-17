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
MODEL_PATH = os.path.join(settings.BASE_DIR, 'gaeyeon', 'dog_breed_classifier.keras')
model = load_model(MODEL_PATH)

CLASS_NAMES = ['chihuahua', 'newfoundland', 'chesapeakebayretriever', 'saluki', 'pomeranian',
               'germanshepherd', 'samoyed', 'boxer', 'australianterrier', 'kuvasz',
               'welshspringerspaniel', 'hound', 'yorkshireterrier', 'greatpyrenees', 'borzoi',
               'staffordshirebullterrier', 'cockerspaniel', 'standardschnauzer', 'papillon', 'poodle',
               'retriever', 'miniaturepinscher', 'borderterrier', 'irishsetter', 'saintbernard',
               'oldenglishsheepdog', 'basenji', 'irishterrier', 'englishsetter', 'kerryblueterrier',
               'greatdane', 'bloodhound', 'rottweiler', 'beagle', 'scottishdeerhound',
               'affenpinscher', 'pug', 'westhighlandwhiteterrier', 'miniatureschnauzer', 'keeshond',
               'siberianhusky', 'giantschnauzer', 'irishwaterspaniel', 'frenchbulldog', 'gordonsetter',
               'vizsla', 'bordercollie', 'weimaraner', 'rhodesianridgeback', 'shihtzu',
               'irishwolfhound'
               ]

csv_path = os.path.join(settings.BASE_DIR, 'gaeyeon', 'dog_breeds_info.csv')
df = pd.read_csv(csv_path)


@csrf_exempt
def upload_view(request):
    if request.method == 'POST':
        try:
            # 1. 텍스트 입력 처리
            if 'breed_text' in request.POST and request.POST['breed_text']:
                breed_text = request.POST['breed_text'].strip().lower()

                # DB에 견종 정보가 있는지 확인
                if breed_text not in df['Breed'].str.lower().values:
                    return JsonResponse({
                        'success': False,
                        'message': f"'{breed_text}' 견종의 정보는 아직 준비되지 않았습니다."
                    })

                request.session['predicted_breed'] = breed_text
                return JsonResponse({'success': True, 'redirect': '/survey/'})

            # 2. 이미지 입력 처리
            elif 'fileInput' in request.FILES:
                image_file = request.FILES['fileInput']

                # 모델이 384x384 크기를 기대하므로 (384, 384)로 수정
                img = Image.open(image_file)
                img = img.resize((384, 384))
                img_array = image.img_to_array(img)

                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.

                prediction = model.predict(img_array)
                confidence = np.max(prediction) * 100
                predicted_breed = CLASS_NAMES[np.argmax(prediction)].lower()

                # 예측 실패 (확률 30% 미만)
                if confidence < 30.5:
                    return JsonResponse({
                        'success': False,
                        'message': f"견종을 확신할 수 없습니다. (예측 확률: {confidence:.2f}%)",
                        'confidence': f"{confidence:.2f}%"
                    })

                # DB에 견종 정보가 있는지 확인
                if predicted_breed not in df['Breed'].str.lower().values:
                    return JsonResponse({
                        'success': False,
                        'message': f"'{predicted_breed}' 견종의 정보는 아직 준비되지 않았습니다."
                    })

                request.session['predicted_breed'] = predicted_breed
                return JsonResponse({
                    'success': True,
                    'redirect': '/survey/',
                    'breed': predicted_breed,
                    'confidence': f"{confidence:.2f}%"
                })

            else:
                return JsonResponse({
                    'success': False,
                    'message': '입력 누락: image나 breed_text가 필요합니다.'
                }, status=400)

        except Exception as e:
            # 모든 종류의 서버 내부 에러를 캐치
            return JsonResponse({'success': False, 'message': f'서버 내부 오류: {str(e)}'}, status=500)

    return render(request, 'upload.html')


def survey_view(request):
    breed = request.session.get('predicted_breed')
    if not breed:
        return redirect('upload_view')

    if request.method == 'POST':
        # q2부터 수집하도록 range(2, 22)로 수정
        q = {f'q{i}': request.POST.get(f'q{i}', '') for i in range(2, 22)}

        # --- ### 1순위: 논리 오류(재설문) 검사 ### ---
        # 불합격 조건을 검사하기 전에, 논리적 모순부터 검사합니다.
        warning_conditions = [
            q.get('q5') == '50만원 이상' and q.get('q6') != '600만원 이상',
            q.get('q5') != '50만원 이상' and q.get('q6') == '600만원 이상',
            q.get('q7') in ['20만원~25만원 미만', '25만원 이상'] and q.get('q8') not in ['240만원~260만원 미만', '260만원 이상'],
            q.get('q7') not in ['20만원~25만원 미만', '25만원 이상'] and q.get('q8') in ['240만원~260만원 미만', '260만원 이상'],
            q.get('q2') == '1명' and q.get('q10') in ['2명', '3명 이상'],
            q.get('q2') == '2명' and q.get('q10') == '3명 이상',
            q.get('q2') == '1명' and q.get('q11') in ['2명', '3명 이상'],
            q.get('q2') == '2명' and q.get('q11') == '3명 이상'
        ]

        if any(warning_conditions):
            if request.session.get('survey_warning_once'):
                # 논리 오류로 2번째 걸리면 '불합격' 처리
                return render(request, 'fail.html', {'fail_reason': '설문 문항 내 논리적 오류'})
            else:
                # 1번째 걸리면 '재설문' 요청
                request.session['survey_warning_once'] = True
                # breed 값을 템플릿으로 다시 전달해야 합니다.
                return render(request, 'survey.html', {'warning': '설문을 진실 되게 작성 하세요.', 'breed': breed})
        # --- ###################################### ---

        # --- 2순위: 불합격 조건 검사 (논리 오류 통과 시) ---
        breed_info_df = df[df['Breed'].str.lower() == breed.lower()]
        if breed_info_df.empty:
            return render(request, 'fail.html', {'fail_reason': '해당 견종의 정보를 찾을 수 없습니다.'})
        breed_info = breed_info_df.iloc[0]

        # 1. Height (in) 처리 (예: "21-24" -> 22.5)
        height_str = str(breed_info['Height (in)'])
        if '-' in height_str:
            parts = height_str.split('-')
            height = (int(parts[0]) + int(parts[1])) / 2
        else:
            height = int(height_str)

        # 2. Longevity (yrs) 처리 (예: "10-12" -> 11)
        longevity_str = str(breed_info['Longevity (yrs)'])
        if '-' in longevity_str:
            parts = longevity_str.split('-')
            longevity = (int(parts[0]) + int(parts[1])) / 2
        else:
            longevity = int(longevity_str)

        # 3. 나머지 정보 로드
        fur_color = breed_info['Fur Color']
        eye_color = breed_info['Color of Eyes']
        traits_str = breed_info['Character Traits']

        # (불합격 조건 계산)
        dog_actual_traits = [trait.strip() for trait in traits_str.split(',')]
        extrovert_keywords = ['active', 'affectionate', 'athletic', 'brave', 'confident', 'curious', 'energetic',
                              'friendly', 'playful', 'social']
        introvert_keywords = ['calm', 'charming']

        is_extrovert = any(trait in extrovert_keywords for trait in dog_actual_traits)
        is_introvert = any(trait in introvert_keywords for trait in dog_actual_traits)

        dog_tendency = []
        if is_extrovert:
            dog_tendency.append('extrovert')
        if is_introvert:
            dog_tendency.append('introvert')
        if not dog_tendency:
            dog_tendency = ['extrovert', 'introvert']

        fail_traits = q.get('q21') not in dog_tendency

        economic_weak = (q.get('q5') != '50만원 이상' or q.get('q6') != '600만원 이상'
                         or q.get('q7') not in ['20만원~25만원 미만', '25만원 이상']
                         or q.get('q8') not in ['240만원~260만원 미만', '260만원 이상'])

        environment_weak = (q.get('q4') == '30분 이상' or q.get('q9') != '해당X'
                            or q.get('q10') != '해당X' or q.get('q11') != '해당X'
                            or q.get('q12') == '아니오')

        minidog_walk = (q.get('q19') in ['miniminidog', 'minidog']
                        and q.get('q13') == '30분~50분 미만'
                        and q.get('q14') == '1km~3km 미만')
        middledog_walk = (q.get('q19') == 'middledog'
                          and q.get('q13') == '약 1시간'
                          and q.get('q14') == '3km~5km 미만')
        bigdog_walk = (q.get('q19') in ['bigdog', 'bigbigdog']
                       and q.get('q13') == '1시간 이상'
                       and q.get('q14') == '5km~8km 미만')

        fail_q15 = (q.get('q15') != '있음')
        fail_q16 = q.get('q16') in ['복수의 반려견', '대가족 생활']

        # 'minido_walk' 오타 수정 -> 'minidog_walk'
        irresponsible = fail_q15 or fail_q16 or not (minidog_walk or middledog_walk or bigdog_walk)

        size_map = {
            'miniminidog': height > 11,
            'minidog': not (12 <= height <= 15),
            'middledog': not (16 <= height <= 21),
            'bigdog': not (22 <= height <= 27),
            'bigbigdog': height < 28
        }
        fail_q19 = size_map.get(q.get('q19'), False)

        fail_q20 = False
        if q.get('q20') == '10년 미만':
            fail_q20 = longevity >= 10
        elif q.get('q20') == '10년~13년 미만':
            fail_q20 = not (10 <= longevity < 13)
        elif q.get('q20') == '13년 이상':
            fail_q20 = longevity < 13

        fail_q17 = q.get('q17') not in eye_color
        fail_q18 = q.get('q18') not in fur_color

        trait_unmatch = fail_q17 or fail_q18 or fail_q19 or fail_q20 or fail_traits

        # (불합격 조건 검사)
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

        # --- 3순위: 최종 통과 ---
        # 모든 검사(논리, 불합격)를 통과한 경우
        return render(request, 'pass.html')

    # GET 요청 시
    return render(request, 'survey.html', {'breed': breed})