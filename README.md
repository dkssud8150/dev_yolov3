# 가상환경 설정 (using virtualenv)


## 설치 및 가상환경 생성

```bash
pip install virtualenv

virtualenv dev --python=python3.8
```

이 때 3.8 interpreter가 없다고 뜬다면, 3.8 python이 안깔려 있는 것이므로 3.8버전을 깔거나 다른 버전으로 실행한다.

## 가상환경 활성화

```bash
source dev/Scripts/activate
```

## 필요한 패키지 설치

```bash
pip install numpy
```

## 나가기

```bash
deactivate
```

<br>

## 자신이 설치한 패키지를 저장하기

```bash
pip freeze > requirements.txt
```

## 다시 설치

```bash
pip install -r requirements.txt
```

<br>
