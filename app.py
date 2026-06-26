"""
Caju — Agente de Inteligência de Conteúdo
Hackathon MKT.IA · Junho 2026

Para rodar localmente:
    streamlit run app.py

Para compartilhar via Streamlit Community Cloud:
    1. Suba este projeto para um repositório GitHub
    2. Acesse share.streamlit.io e conecte o repositório
    3. Compartilhe o link gerado com o time
"""

import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ── Configuração da página ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agente de Conteúdo · Caju",
    page_icon="🥤",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Estilo visual (identidade Caju) ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Header principal */
.caju-header {
    background: linear-gradient(135deg, #E80537 0%, #FF7500 100%);
    border-radius: 16px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 24px;
}
.caju-header h1 { font-size: 28px; font-weight: 700; margin: 0; }
.caju-header p  { font-size: 15px; opacity: 0.88; margin: 6px 0 0; }

/* Badge */
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Card de resultado */
.result-card {
    background: #FFFAF3;
    border: 1.5px solid #E80537;
    border-radius: 12px;
    padding: 24px;
    white-space: pre-wrap;
    line-height: 1.75;
    font-size: 14px;
    color: #1F1F1F;
}

/* Fontes */
.fontes {
    background: #F3F3F3;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #434343;
    margin-top: 12px;
}

/* Botão primário customizado */
div.stButton > button {
    background: #E80537;
    color: white;
    border: none;
    border-radius: 999px;
    padding: 12px 32px;
    font-weight: 700;
    font-size: 15px;
    width: 100%;
    transition: background 0.2s;
}
div.stButton > button:hover {
    background: #DC0320;
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #FFFAF3;
}
</style>
""", unsafe_allow_html=True)

# ── Sistema de prompt (voz da Caju) ────────────────────────────────────────────
SYSTEM_PROMPT = """Você é o Agente de Inteligência de Conteúdo da Caju — a maior plataforma de multibenefícios flexíveis do Brasil.

TOM DE VOZ DA CAJU:
- Direto e humano. Zero jargão corporativo.
- Próximo, como alguém que entende de RH e de negócios ao mesmo tempo.
- Confiante, nunca arrogante.
- Brasileiro de verdade — não force anglicismos.
- Use dados concretos. Nunca seja vago.

SOBRE A CAJU:
- Plataforma de multibenefícios flexíveis líder no Brasil
- Produtos: Caju Flex (VA/VR flexível), Caju Alimentação, Caju Mobilidade, Caju Saúde
- Aceito em mais de 2 milhões de estabelecimentos e 109 países
- Mais de 6 mil empresas clientes
- Pesquisa anual: Panorama do RH

DADOS DO PANORAMA DO RH:
- 73% dos profissionais consideram benefícios flexíveis decisivos na escolha do emprego
- Empresas com benefícios flexíveis têm 40% menos rotatividade
- 67% dos colaboradores nunca tiveram liberdade para escolher seus benefícios
- 82% dos gestores citam rotatividade como principal desafio

INSTRUÇÕES:
1. Use os trechos de conteúdo fornecidos como base
2. Personalize com o segmento e porte do lead
3. Cite dados do Panorama do RH quando reforçar o argumento
4. Todo material deve terminar com CTA claro
5. Ao final, adicione '📎 Fontes utilizadas:' com os títulos dos artigos usados
6. Em e-mails, inclua linha de assunto sugerida no início"""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuração")

    api_key = st.text_input(
        "Chave da OpenAI",
        type="password",
        placeholder="sk-proj-...",
        help="Acesse platform.openai.com → API Keys"
    )

    chroma_path = st.text_input(
        "Caminho da base vetorial",
        value="./data/chroma",
        help="Pasta onde está o chroma.sqlite3"
    )

    st.divider()
    st.markdown("### 📊 Status da base")

    if api_key and os.path.exists(chroma_path):
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
            vs = Chroma(collection_name="caju_content", embedding_function=embeddings, persist_directory=chroma_path)
            total = vs._collection.count()
            st.success(f"✅ {total} chunks indexados")
        except Exception as e:
            st.error(f"Erro: {str(e)[:80]}")
    else:
        st.info("Configure a chave e o caminho da base para verificar.")

    st.divider()
    st.markdown("""
    **🥤 Agente de Conteúdo Caju**
    Hackathon MKT.IA · Junho 2026
    Time: Cecilia · Ana Luisa · Israel · Jonatan · Pedro
    """)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="caju-header">
    <div class="badge">Hackathon MKT.IA · 2026</div>
    <h1>🥤 Agente de Conteúdo</h1>
    <p>Descreva o perfil do lead e gere materiais alinhados à voz da Caju em segundos.</p>
</div>
""", unsafe_allow_html=True)

# ── Formulário ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    segmento = st.text_input("🏢 Segmento do lead", placeholder="Ex: construção civil, varejo, tech...")
    formato = st.selectbox("📄 Formato do material", [
        "E-mail comercial",
        "One pager",
        "Roteiro de apresentação (PPT)",
        "Artigo de blog",
        "Proposta comercial",
    ])

with col2:
    porte = st.selectbox("👥 Porte da empresa", [
        "Micro (até 9 funcionários)",
        "Pequena (10 a 49 funcionários)",
        "Média (50 a 499 funcionários)",
        "Grande (500+ funcionários)",
    ])
    tom = st.selectbox("🎙️ Tom", [
        "Consultivo e próximo",
        "Direto e objetivo",
        "Educacional e didático",
        "Urgência e oportunidade",
    ])

dores = st.text_area(
    "💬 Dores / necessidades do lead",
    placeholder="Ex: alta rotatividade de funcionários e dificuldade em controlar custos com benefícios...",
    height=100,
)

gerar = st.button("✨ Gerar material")

# ── Geração ────────────────────────────────────────────────────────────────────
if gerar:
    if not api_key:
        st.error("Configure sua chave da OpenAI na barra lateral.")
    elif not segmento:
        st.error("Informe o segmento do lead.")
    elif not dores:
        st.error("Descreva as dores ou necessidades do lead.")
    elif not os.path.exists(chroma_path):
        st.error(f"Base vetorial não encontrada em '{chroma_path}'. Verifique o caminho.")
    else:
        with st.spinner("🔍 Buscando conteúdo relevante na base..."):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
                vectorstore = Chroma(
                    collection_name="caju_content",
                    embedding_function=embeddings,
                    persist_directory=chroma_path,
                )

                query = f"{segmento} {dores} benefícios flexíveis RH"
                chunks = vectorstore.similarity_search_with_score(query, k=8)

                if not chunks:
                    st.warning("Nenhum conteúdo encontrado. Verifique se a base foi indexada.")
                else:
                    # Separa chunks por tipo de conteúdo
                    chunks_conteudo = [(c, s) for c, s in chunks if c.metadata.get("content_type") != "email_template"]
                    chunks_email    = [(c, s) for c, s in chunks if c.metadata.get("content_type") == "email_template"]

                    # Contexto para o LLM (usa todos)
                    contexto = "\n\n---\n\n".join([
                        f"Fonte: {c.metadata.get('title','Sem título')} ({c.metadata.get('source_name','')})\n\n{c.page_content[:800]}"
                        for c, _ in chunks
                    ])

                    # Fontes com URL (deduplicadas por URL)
                    vistas = set()
                    fontes_com_link = []
                    for c, score in chunks_conteudo:
                        url   = c.metadata.get("source", "")
                        title = c.metadata.get("title", "Sem título")
                        src   = c.metadata.get("source_name", "")
                        rel   = round((1 - score) * 100, 1)
                        if url and url not in vistas and not url.startswith("email_templates"):
                            vistas.add(url)
                            fontes_com_link.append({"title": title, "url": url, "source": src, "relevancia": rel})

                    # Artigos similares: próximos 4 da busca que não foram usados diretamente
                    similares_query = f"{segmento} {dores} benefícios RH colaboradores"
                    similares_raw   = vectorstore.similarity_search_with_score(similares_query, k=12)
                    vistas_sim = {f["url"] for f in fontes_com_link}
                    artigos_similares = []
                    for c, score in similares_raw:
                        url   = c.metadata.get("source", "")
                        title = c.metadata.get("title", "Sem título")
                        src   = c.metadata.get("source_name", "")
                        rel   = round((1 - score) * 100, 1)
                        if url and url not in vistas_sim and not url.startswith("email_templates") and len(artigos_similares) < 4:
                            vistas_sim.add(url)
                            artigos_similares.append({"title": title, "url": url, "source": src, "relevancia": rel})

            except Exception as e:
                st.error(f"Erro ao acessar a base: {str(e)}")
                st.stop()

        with st.spinner(f"✍️ Gerando {formato.lower()}..."):
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0.75, max_tokens=2000)

                user_prompt = f"""Gere um {formato} com tom {tom.lower()} para um lead com o seguinte perfil:

- Segmento: {segmento}
- Porte: {porte}
- Dores e necessidades: {dores}

Use os trechos de conteúdo abaixo como base:

{contexto}

Produza o material completo, pronto para uso. Ao final, inclua '📎 Fontes utilizadas:' com os títulos dos artigos que você usou."""

                resposta = llm.invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ])

                resultado = resposta.content

            except Exception as e:
                st.error(f"Erro ao gerar material: {str(e)}")
                st.stop()

        # ── Exibe resultado ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### ✅ {formato} gerado")

        st.markdown(f'<div class="result-card">{resultado}</div>', unsafe_allow_html=True)

        st.download_button(
            label="⬇️ Baixar como .txt",
            data=resultado,
            file_name=f"caju_{segmento.lower().replace(' ','_')}_{formato.lower().replace(' ','_')}.txt",
            mime="text/plain",
        )

        # ── Fontes utilizadas (com links clicáveis) ──────────────────────────
        if fontes_com_link:
            with st.expander("📎 Fontes utilizadas na geração", expanded=True):
                for f in fontes_com_link:
                    if f["url"].startswith("http"):
                        st.markdown(
                            f'- [{f["title"]}]({f["url"]}) — *{f["source"]}* '
                            f'<span style="font-size:11px;color:#999;">({f["relevancia"]}% relevante)</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f'- {f["title"]} — *{f["source"]}*')

        # ── Artigos similares sugeridos ───────────────────────────────────────
        if artigos_similares:
            with st.expander("🔗 Artigos relacionados — leitura complementar"):
                st.markdown(
                    "<small>Outros conteúdos da base que podem enriquecer seu material:</small>",
                    unsafe_allow_html=True,
                )
                for a in artigos_similares:
                    if a["url"].startswith("http"):
                        st.markdown(
                            f'- [{a["title"]}]({a["url"]}) — *{a["source"]}*',
                        )
                    else:
                        st.markdown(f'- {a["title"]} — *{a["source"]}*')
