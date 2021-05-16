from faker import Faker
import pandas as pd
import click
import numpy as np

faker = Faker()
Faker.seed(0)


def create_fake_data(size=100):
    rows = [{
         'Age': faker.pyint(0, 100),
         'Number of sexual partners': faker.pyint(0, 10),
         'First sexual intercourse': faker.pyint(13, 50),
         'Num of pregnancies': faker.pyint(0, 10),
         'Smokes': faker.pyint(0, 1),
         'Smokes (years)': faker.pyint(0, 80),
         'Smokes (packs/year)': faker.pyint(0, 300),
         'Hormonal Contraceptives': faker.pyint(0, 1),
         'Hormonal Contraceptives (years)': faker.pyint(0, 50),
         'IUD': faker.pyint(0, 1),
         'IUD (years)': faker.pyint(0, 50),
         'STDs': faker.pyint(0, 1),
         'STDs (number)': faker.pyint(0, 15),
         'STDs:condylomatosis': faker.pyint(0, 1),
         'STDs:cervical condylomatosis': faker.pyint(0, 1),
         'STDs:vaginal condylomatosis': faker.pyint(0, 1),
         'STDs:vulvo-perineal condylomatosis': faker.pyint(0, 1),
         'STDs:syphilis': faker.pyint(0, 1),
         'STDs:pelvic inflammatory disease': faker.pyint(0, 1),
         'STDs:genital herpes': faker.pyint(0, 1),
         'STDs:molluscum contagiosum': faker.pyint(0, 1),
         'STDs:AIDS': faker.pyint(0, 1),
         'STDs:HIV': faker.pyint(0, 1),
         'STDs:Hepatitis B': faker.pyint(0, 1),
         'STDs:HPV': faker.pyint(0, 1),
         'STDs: Number of diagnosis': faker.pyint(0, 15),
         'STDs: Time since first diagnosis': faker.pyint(0, 80),
         'STDs: Time since last diagnosis': faker.pyint(0, 80),
         'Dx:Cancer': faker.pyint(0, 1),
         'Dx:CIN': faker.pyint(0, 1),
         'Dx:HPV': faker.pyint(0, 1),
         'Dx': faker.pyint(0, 1),
         'Hinselmann': faker.pyint(0, 1),
         'Schiller': faker.pyint(0, 1),
         'Citology': faker.pyint(0, 1),
         'Biopsy': faker.pyint(0, 1),
    } for _ in range(size)]

    fake_data = pd.DataFrame(rows)
    mask_features = np.random.randint(0, 10, size=(fake_data.shape[0], fake_data.shape[1] - 1))
    mask_target = np.ones((fake_data.shape[0], 1))
    mask = np.hstack((mask_features, mask_target)).astype(np.bool)

    fake_data[~mask] = np.nan
    for column in fake_data.columns:
        fake_data[column] = fake_data[column].fillna('?')

    fake_data.to_csv("data/train_data_sample.csv", index=False)


@click.command(name="create_fake_data")
@click.argument("size")
def create_fake_data_command(size: str):
    create_fake_data(int(size))


if __name__ == "__main__":
    create_fake_data_command()