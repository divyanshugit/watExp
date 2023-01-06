import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--f_lang",
        type=str,
        required=True,
        help="Enter the any Indic Language present in Samanatar Dataset",
    )
    parser.add_argument(
        "--s_lang",
        type=str,
        required=True,
        help="Enter the any Indic Language present in Samanatar Dataset",
    )
    
    LANGUAGES = {"Assamese": "as", "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn",
                "Malayalam": "ml", "Marathi": "mr", "Odia": "or", "Punjabi": "pa", "Tamil": "ta", "Telugu": "te", "English": "en"}

    args = parser.parse_args()

    first_language  = args.f_lang
    second_language = args.s_lang
    f_lang = LANGUAGES[first_language]
    s_lang = LANGUAGES[second_language]

    firstSet = f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/samanatar/v2/en-{f_lang}"
    secondSet = f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/samanatar/v2/en-{s_lang}"

    # firstSetSI, firstSetSII, secondSetSI, secondSetSII subject to change with a better variable name

    with open(f'{firstSet}/train.en', 'r',encoding='utf-8', errors='ignore') as f:
        firstSetSI = f.readlines()

    with open(f'{firstSet}/train.{f_lang}', 'r',encoding='utf-8', errors='ignore') as f:
        firstSetSII = f.readlines()

    with open(f'{secondSet}/train.en', 'r',encoding='utf-8', errors='ignore') as f:
        secondSetSI = f.readlines()

    with open(f'{secondSet}/train.{s_lang}', 'r',encoding='utf-8', errors='ignore') as f:
        secondSetSII = f.readlines()

    print(f"firstSetSI: {len(firstSetSI)}, Unique firstSetSI: {len(set(firstSetSI))}")
    print(f"firstSetSII: {len(firstSetSII)}, Unique firstSetSII: {len(set(firstSetSII))}")
    print(f"secondSetSI: {len(secondSetSI)}, Unique secondSetSI: {len(set(secondSetSI))}")
    print(f"secondSetSII: {len(secondSetSII)}, Unique secondSetSII: {len(set(secondSetSII))}")

    # print(len(list(set(len(firstSetSI)) & set(len(secondSetSI)))))

    intersect = set(firstSetSI).intersection(secondSetSI)
    # print(intersect[:25])
    print(len(intersect))


    df1 = pd.DataFrame({
        "English":firstSetSI,
        f"{first_language}":firstSetSII
        })
    df2 = pd.DataFrame({
        "English":secondSetSI,
        f"{second_language}":secondSetSII
        })

    df1.English = df1.English.apply(lambda x: x.replace('\n', ''))
    df1[f'{first_language}'] = df1[f'{first_language}'].apply(lambda x: x.replace('\n',''))
    df2.English = df2.English.apply(lambda x: x.replace('\n',''))
    df2[f'{second_language}'] = df2[f'{second_language}'].apply(lambda x: x.replace('\n',''))


    df = pd.merge(df1, df2, on='English')
    print(df.shape)
    df.to_csv(f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/parallelData/withDuplicates/en{f_lang}{s_lang}_wd.csv",index=False)
    df.drop_duplicates(subset=['English'], inplace=True)
    print(df.shape)
    df.to_csv(f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/parallelData/cleanData/en{f_lang}{s_lang}.csv",index=False)

if __name__ == '__main__':
    main()


