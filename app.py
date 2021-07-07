import streamlit as st
import plotly.express as px
import pandas as pd
import pickle
import os
import base64
from io import BytesIO
from datetime import datetime


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    val = to_excel(df)
    b64 = base64.b64encode(val)
    date_now = datetime.utcnow()
    file_name = f'data_resultado-{date_now.strftime("%Y%m%d%H%M%S")}.xlsx'
    link_download = f""" <a href="data:application/octet-stream;base64,{b64.decode()}" download="{file_name}">Download xlsx file</a> """
    return link_download


def plot_graph(df_graph):
    fig = px.bar(
        df_graph,
        x='Labels',
        y='Descrição',
        # text='Text test',
        title='Test',
        labels={
              "Labels": "Labels",
            "Descrição": 'Número de coisas'
        },
        # width=1400,
        height=500
    )
    return fig


def main(classificador):
    st.title('Model')
    process_file = st.file_uploader(
        "Faça o upload do arquivo no campo abaixo.",
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

        with st.empty():
            st.write('Fazendo as predições ...')
            df['Labels'] = classificador.predict(
                df["Descrição"].astype("unicode"))
            st.write('Predições feitas com sucesso !!!')

        st.dataframe(df.head(20))

        df_graph = df.groupby(['Labels'], as_index=False)['Descrição'].count()
        df_graph.sort_values(by=['Descrição'], inplace=True, ascending=False)
        print(df_graph)
        st.plotly_chart(plot_graph(df_graph), use_container_width=True)
        st.text('Gerando link para download ...')
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
        st.success('Link gerado com sucesso.')


if __name__ == '__main__':
    classificador = pickle.load(open("modelo_final.pkl", "rb"))
    main(classificador)
