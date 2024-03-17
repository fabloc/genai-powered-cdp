from mimesis import Person
from mimesis import Address
from mimesis.enums import Gender
from mimesis import Datetime
from mimesis.enums import Locale
from mimesis.random import Random
from mimesis.locales import Locale
from mimesis import Generic
import uuid
import pandas as pd
import random
from mimesis.enums import TimestampFormat

mu = 40
sigma = 15

def create_rows_mimesis(person, address, country, num=1):
    output = [{ "id": uuid.uuid4(),
                "first_name":(first_name := person.first_name(Gender(gender := random.choice(['male', 'female'])))),
                "last_name": (last_name := person.last_name()),
                "gender": gender,
                "age": max(min(int(random.gauss(mu, sigma)), 100), 0),
                "state": address.state(),
                "street_address": address.street_number() + ', ' + address.street_name(),
                "postal_code": address.postal_code(),
                "city": address.city(),
                "country": country,
                "email": person.email(domains=["gmail.com", "yahoo.com", "hotmail.com", "aol.com", "hotmail.fr", "msn.com", "yahoo.fr", "wanadoo.fr", "orange.fr", "comcast.net", "live.com" ], unique=True),
                "created_at": datetime.timestamp(start=2020, end=2023, fmt=TimestampFormat.ISO_8601)} for x in range(num)]
    return output

country_locale_map = {
    "Czech Republic": "cs",
    "Denmark": "da",
    "Germany": "de",
    "Greece": "el",
    "United States": "en",
    "Spain": "es",
    "Estonia": "et",
    "Iran": "fa",
    "Finland": "fi",
    "France": "fr",
    "Croatia": "hr",
    "Hungary": "hu",
    "Iceland": "is",
    "Italy": "it",
    "Japan": "ja",
    "Kazakhstan": "kk",
    "South Korea": "ko",
    "Netherlands": "nl",
    "Norway": "no",
    "Poland": "pl",
    "Portugal": "pt",
    "Russia": "ru",
    "Slovakia": "sk",
    "Sweden": "sv",
    "Turkey": "tr",
    "Ukraine": "uk",
    "China": "zh"
}
mimesis_list = []
datetime = Datetime()
for country, locale in country_locale_map.items():
    person = Person(locale)
    address = Address(locale)
    df_mimesis = pd.DataFrame(create_rows_mimesis(person, address, country, 3700000))
    df_mimesis.to_csv('users-' + locale + '.csv', sep=',', index=False, encoding='utf-8')