import streamlit as st
import pandas as pd
import pickle
import os


def main(classificador):
    st.title('Model test')
    process_file = st.file_uploader(
        "Faça o upload dos arquivos no campo abaixo.",
        type=["csv", "xlsx"],
        accept_multiple_files=False
    )

    print(process_file)
    print(os.environ.get('TOKEN'))
    if process_file != None:
        if process_file.name.endswith('.csv'):
            df = pd.read_csv(
                process_file, header=0, skip_blank_lines=True, skipinitialspace=True, encoding='latin-1')

        elif process_file.name.endswith('.xlsx'):
            df = pd.read_excel(
                process_file, engine="openpyxl")

        st.info('Fazendo as predições ...')
        df['Labels'] = classificador.predict(df["Descrição"].astype("unicode"))

        st.success('Predições feitas com sucesso.')

        st.dataframe(df.head(20))


if __name__ == '__main__':
    classificador = pickle.load(open("modelo_final.pkl", "rb"))
    main(classificador)
