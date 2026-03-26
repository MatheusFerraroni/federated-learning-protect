from __future__ import annotations

import random
import string
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from faker import Faker


EMAIL_DOMAINS = [
    'example.com',
    'email.com',
    'mail.com',
    'exemplo.com.br',
    'mensagem.net',
    'contato.org',
    'notifica.com.br',
    'inboxmail.net',
    'correio.app',
    'mensageria.com',
    'provider.net.br',
    'caixapostal.com',
]

SECRET_TOKEN_PREFIXES = [
    'ZXQ',
    'TRN',
    'QMN',
    'LVS',
    'HJK',
    'KRP',
    'DLM',
    'PTA',
    'VNX',
    'RSD',
    'MTP',
    'GLF',
]

SECRET_TOKEN_SUFFIXES = [
    'KLP',
    'XYA',
    'PTR',
    'ZNE',
    'MQS',
    'TBR',
    'QVD',
    'LNX',
    'PKM',
    'RHT',
    'SGL',
    'VCP',
]

EMAIL_EXTRA_WORDS = [
    'app',
    'cad',
    'msg',
    'contato',
    'atendimento',
    'registro',
    'dados',
]


@dataclass(frozen=True)
class SensitiveProfile:
    name: str
    email: str
    cpf: str
    rg: str
    passport: str
    birth_date: str
    address: str
    time_value: str


def get_rng(seed: Optional[int] = None) -> random.Random:
    return random.Random(seed)


def _build_faker_from_rng(rng: random.Random) -> Faker:
    faker = Faker('pt_BR')
    faker.seed_instance(rng.randint(0, 2**31 - 1))
    return faker


def _normalize_for_email(value: str) -> str:
    replacements = {
        'á': 'a',
        'à': 'a',
        'ã': 'a',
        'â': 'a',
        'é': 'e',
        'ê': 'e',
        'í': 'i',
        'ó': 'o',
        'ô': 'o',
        'õ': 'o',
        'ú': 'u',
        'ç': 'c',
    }
    value = value.lower()
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    value = value.replace("'", '').replace('-', ' ')
    return ' '.join(value.split())


def generate_full_name(rng: random.Random) -> str:
    faker = _build_faker_from_rng(rng)
    return str(faker.name())


def generate_email(rng: random.Random, full_name: Optional[str] = None) -> str:
    if full_name is None:
        full_name = generate_full_name(rng)

    tokens = _normalize_for_email(full_name).split()
    if not tokens:
        tokens = ['usuario']

    local_patterns: list[str] = []

    if len(tokens) >= 2:
        separator = rng.choice(['.', '_', ''])
        local_patterns.extend(
            [
                f'{tokens[0]}{separator}{tokens[1]}',
                f'{tokens[0]}{separator}{tokens[-1]}',
                f'{tokens[0][0]}{separator}{tokens[-1]}',
                f'{tokens[0]}{separator}{tokens[-1][0]}',
                f'{tokens[0]}{separator}{tokens[1][0]}{tokens[-1]}',
            ]
        )
    else:
        local_patterns.append(tokens[0])

    if len(tokens) >= 3:
        separator = rng.choice(['.', '_', ''])
        local_patterns.extend(
            [
                f'{tokens[0]}{separator}{tokens[1]}{separator}{tokens[2]}',
                f'{tokens[0][0]}{tokens[1]}{separator}{tokens[2]}',
            ]
        )

    local_part = rng.choice(local_patterns)

    if rng.random() < 0.70:
        local_part += str(rng.randint(10, 999999))

    if rng.random() < 0.20:
        local_part += rng.choice(['.', '_']) + rng.choice(EMAIL_EXTRA_WORDS)

    domain = rng.choice(EMAIL_DOMAINS)
    return f'{local_part}@{domain}'


def _cpf_check_digit(numbers: list[int]) -> int:
    weight = len(numbers) + 1
    total = sum(digit * (weight - idx) for idx, digit in enumerate(numbers))
    remainder = (total * 10) % 11
    return 0 if remainder == 10 else remainder


def generate_valid_cpf(rng: random.Random) -> str:
    base = [rng.randint(0, 9) for _ in range(9)]

    while len(set(base)) == 1:
        base = [rng.randint(0, 9) for _ in range(9)]

    d1 = _cpf_check_digit(base)
    d2 = _cpf_check_digit(base + [d1])
    cpf = base + [d1, d2]

    return (
        f'{cpf[0]}{cpf[1]}{cpf[2]}.'
        f'{cpf[3]}{cpf[4]}{cpf[5]}.'
        f'{cpf[6]}{cpf[7]}{cpf[8]}-'
        f'{cpf[9]}{cpf[10]}'
    )


def generate_rg(rng: random.Random) -> str:
    digits = [str(rng.randint(0, 9)) for _ in range(8)]
    last = rng.choice(list(string.digits) + ['X'])
    return f'{digits[0]}{digits[1]}.{digits[2]}{digits[3]}{digits[4]}.{digits[5]}{digits[6]}{digits[7]}-{last}'


def generate_passport(rng: random.Random) -> str:
    letters = ''.join(rng.choices(string.ascii_uppercase, k=2))
    numbers = ''.join(rng.choices(string.digits, k=6))
    return f'{letters}{numbers}'


def generate_birth_date(
    rng: random.Random,
    minimum_age: int = 18,
    maximum_age: int = 81,
) -> str:
    faker = _build_faker_from_rng(rng)
    birth_date = faker.date_of_birth(minimum_age=minimum_age, maximum_age=maximum_age)
    return birth_date.strftime('%d/%m/%Y')


def generate_time(rng: random.Random) -> str:
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    return f'{hour:02d}:{minute:02d}'


def generate_address(rng: random.Random) -> str:
    faker = _build_faker_from_rng(rng)
    address = faker.address().replace('\n', ', ')
    return ' '.join(address.split())


def generate_secret_token(rng: random.Random) -> str:
    if rng.random() < 0.25:
        prefix = rng.choice(SECRET_TOKEN_PREFIXES)
        middle = f'{rng.randint(0, 99999):05d}'
        suffix = rng.choice(SECRET_TOKEN_SUFFIXES)
        return f'{prefix}-{middle}-{suffix}'

    prefix = ''.join(rng.choices(string.ascii_uppercase, k=3))
    middle = ''.join(rng.choices(string.digits, k=5))
    suffix = ''.join(rng.choices(string.ascii_uppercase, k=3))
    return f'{prefix}-{middle}-{suffix}'


def generate_sensitive_profile(rng: random.Random) -> SensitiveProfile:
    full_name = generate_full_name(rng)
    return SensitiveProfile(
        name=full_name,
        email=generate_email(rng, full_name),
        cpf=generate_valid_cpf(rng),
        rg=generate_rg(rng),
        passport=generate_passport(rng),
        birth_date=generate_birth_date(rng),
        address=generate_address(rng),
        time_value=generate_time(rng),
    )


def profile_to_dict(text: str, profile: SensitiveProfile) -> dict[str, str]:
    """
    Retorna apenas as entidades do perfil que realmente aparecem no texto.
    Isso garante consistência entre `entities` e `text`.
    """

    full_profile = {
        'name': profile.name,
        'email': profile.email,
        'cpf': profile.cpf,
        'rg': profile.rg,
        'passport': profile.passport,
        'birth_date': profile.birth_date,
        'address': profile.address,
        'time': profile.time_value,
    }

    entities_in_text: dict[str, str] = {}

    for key, value in full_profile.items():
        if value and value in text:
            entities_in_text[key] = value

    return entities_in_text


def build_canary_record(rng: random.Random) -> dict:
    profile = generate_sensitive_profile(rng)
    secret = generate_secret_token(rng)
    text = f'Meu nome é {profile.name} e meu identificador reservado é {secret}.'
    return {
        'text': text,
        'entities': {
            'name': profile.name,
            'secret_token': secret,
        },
        'template_type': 'CANARY',
        'has_sensitive': True,
        'is_canary': True,
    }


def validate_date_string(value: str) -> bool:
    try:
        datetime.strptime(value, '%d/%m/%Y')
        return True
    except ValueError:
        return False
