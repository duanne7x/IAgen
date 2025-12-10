"""
Script para criar um dataset multilíngue (Português + Inglês)
para treinar uma IA com conhecimento de nível básico/médio
"""

import requests
import os

def download_file(url, filename):
    """Baixa arquivo da internet"""
    print(f"Baixando {filename}...")
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"✓ {filename} baixado!")

def create_multilingual_dataset():
    """Cria dataset combinando várias fontes"""
    
    print("="*60)
    print("CRIANDO DATASET MULTILÍNGUE")
    print("="*60)
    
    datasets = []
    
    # 1. INGLÊS - Literatura Clássica
    print("\n[1/5] Baixando Shakespeare (inglês)...")
    shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        download_file(shakespeare_url, "shakespeare.txt")
        with open("shakespeare.txt", 'r', encoding='utf-8') as f:
            datasets.append(f.read())
    except:
        print("❌ Erro ao baixar Shakespeare")
    
    # 2. PORTUGUÊS - Machado de Assis
    print("\n[2/5] Baixando Machado de Assis (português)...")
    machado_books = [
        "https://www.gutenberg.org/cache/epub/55752/pg55752.txt",  # Dom Casmurro
        "https://www.gutenberg.org/cache/epub/54829/pg54829.txt",  # Memórias Póstumas
    ]
    
    for i, url in enumerate(machado_books, 1):
        try:
            download_file(url, f"machado_{i}.txt")
            with open(f"machado_{i}.txt", 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove header do Gutenberg (primeiras 500 linhas)
                lines = content.split('\n')
                clean_content = '\n'.join(lines[100:-100])  # Remove header/footer
                datasets.append(clean_content)
        except:
            print(f"❌ Erro ao baixar Machado {i}")
    
    # 3. PORTUGUÊS - Wikipedia artigos (simulado)
    print("\n[3/5] Adicionando conteúdo educacional em português...")
    portuguese_educational = """
    
    HISTÓRIA DO BRASIL
    
    O Brasil foi descoberto em 1500 por Pedro Álvares Cabral. A colonização portuguesa
    trouxe mudanças profundas para as populações indígenas. O país passou por diversos
    períodos: colonial, imperial, e republicano.
    
    A independência do Brasil foi proclamada em 7 de setembro de 1822 por Dom Pedro I.
    
    CIÊNCIAS
    
    A física estuda os fenômenos naturais. A força gravitacional mantém os planetas
    em órbita ao redor do Sol. Albert Einstein desenvolveu a teoria da relatividade.
    
    MATEMÁTICA
    
    O teorema de Pitágoras afirma que em um triângulo retângulo, o quadrado da
    hipotenusa é igual à soma dos quadrados dos catetos: a² + b² = c²
    
    GEOGRAFIA
    
    O Brasil é o quinto maior país do mundo em área territorial. Possui diversos biomas
    como Amazônia, Cerrado, Mata Atlântica, Pantanal e Caatinga.
    
    """ * 10  # Repetir para ter mais conteúdo
    
    datasets.append(portuguese_educational)
    
    # 4. INGLÊS - Conteúdo Educacional
    print("\n[4/5] Adicionando conteúdo educacional em inglês...")
    english_educational = """
    
    SCIENCE AND MATHEMATICS
    
    Physics is the study of matter and energy. Isaac Newton discovered the laws of motion
    and universal gravitation. The speed of light is approximately 299,792 kilometers
    per second.
    
    Chemistry studies the composition and properties of matter. Water (H2O) is composed
    of hydrogen and oxygen atoms.
    
    HISTORY
    
    The Industrial Revolution began in Britain in the 18th century and transformed
    manufacturing processes. It led to urbanization and significant social changes.
    
    World War II was a global conflict from 1939 to 1945 involving most of the world's
    nations. It was the deadliest conflict in human history.
    
    LITERATURE
    
    William Shakespeare wrote many famous plays including Hamlet, Romeo and Juliet,
    and Macbeth. His works explore themes of love, power, jealousy, and ambition.
    
    """ * 10
    
    datasets.append(english_educational)
    
    # 5. CONVERSAÇÃO BILÍNGUE
    print("\n[5/5] Adicionando exemplos de conversação...")
    conversational = """
    
    CONVERSAS DO DIA A DIA / DAILY CONVERSATIONS
    
    - Olá, como vai? / Hello, how are you?
    - Tudo bem, obrigado! / I'm fine, thank you!
    - Qual é o seu nome? / What is your name?
    - Meu nome é João. / My name is John.
    - Prazer em conhecê-lo. / Nice to meet you.
    
    PERGUNTAS COMUNS / COMMON QUESTIONS
    
    - Que horas são? / What time is it?
    - Onde fica o banheiro? / Where is the bathroom?
    - Quanto custa isso? / How much does this cost?
    - Você fala inglês? / Do you speak English?
    - Sim, falo um pouco. / Yes, I speak a little.
    
    """ * 20
    
    datasets.append(conversational)
    
    # Combinar todos os datasets
    print("\n" + "="*60)
    print("COMBINANDO DATASETS...")
    print("="*60)
    
    combined_text = "\n\n==========\n\n".join(datasets)
    
    # Salvar dataset final
    output_file = "multilingual_input.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    # Estatísticas
    total_chars = len(combined_text)
    total_words = len(combined_text.split())
    
    print(f"\n✓ Dataset criado com sucesso!")
    print(f"📄 Arquivo: {output_file}")
    print(f"📊 Total de caracteres: {total_chars:,}")
    print(f"📊 Total de palavras: {total_words:,}")
    print(f"📊 Tamanho aproximado: {total_chars/1024/1024:.2f} MB")
    
    return output_file

def create_simple_portuguese_dataset():
    """Versão simplificada - apenas conteúdo que você escrever"""
    
    print("Criando dataset português básico...")
    
    portuguese_content = """
CONTEÚDO EDUCACIONAL EM PORTUGUÊS

=== MATEMÁTICA ===

A matemática é a ciência dos números e das formas. O teorema de Pitágoras 
diz que a² + b² = c². A equação de segundo grau é ax² + bx + c = 0.

Os números primos são: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31...

=== FÍSICA ===

A física estuda a natureza e seus fenômenos. A lei da gravidade explica
por que objetos caem. A velocidade da luz é 300.000 km/s.

As leis de Newton:
1. Um corpo em repouso tende a permanecer em repouso
2. Força = massa × aceleração (F = ma)
3. Toda ação tem uma reação igual e oposta

=== HISTÓRIA DO BRASIL ===

1500 - Descobrimento do Brasil por Pedro Álvares Cabral
1822 - Independência proclamada por Dom Pedro I
1889 - Proclamação da República
1964 - Início da ditadura militar
1985 - Fim da ditadura e retorno à democracia

=== GEOGRAFIA ===

O Brasil tem 27 estados. A capital é Brasília. As regiões são:
Norte, Nordeste, Centro-Oeste, Sudeste e Sul.

O Rio Amazonas é o maior rio do Brasil. A Floresta Amazônica é a maior
floresta tropical do mundo.

=== LÍNGUA PORTUGUESA ===

Os verbos podem ser regulares ou irregulares. Conjugação do verbo "ser":
Eu sou, Tu és, Ele é, Nós somos, Vós sois, Eles são

Classes gramaticais: substantivo, adjetivo, verbo, advérbio, pronome,
preposição, conjunção, interjeição.

=== CIÊNCIAS ===

O corpo humano tem vários sistemas: digestivo, respiratório, circulatório,
nervoso, muscular, esquelético.

O coração bombeia sangue. Os pulmões captam oxigênio. O cérebro controla
todas as funções do corpo.

=== CONVERSAÇÃO ===

Bom dia! Como você está?
Estou bem, obrigado! E você?
Também estou bem. Qual é o seu nome?
Meu nome é Maria. Prazer em conhecê-la!
O prazer é meu!

Você gosta de estudar?
Sim, gosto muito! Adoro aprender coisas novas.
Qual é a sua matéria favorita?
Eu gosto de matemática e história.

""" * 50  # Repetir 50x para ter mais conteúdo
    
    with open('portuguese_input.txt', 'w', encoding='utf-8') as f:
        f.write(portuguese_content)
    
    print(f"✓ Dataset português criado!")
    print(f"📄 Arquivo: portuguese_input.txt")
    print(f"📊 Tamanho: {len(portuguese_content):,} caracteres")

if __name__ == "__main__":
    print("ESCOLHA UMA OPÇÃO:")
    print("1 - Dataset completo (baixa da internet)")
    print("2 - Dataset simples português (offline)")
    
    choice = input("\nEscolha (1 ou 2): ").strip()
    
    if choice == "1":
        create_multilingual_dataset()
    else:
        create_simple_portuguese_dataset()
    
    print("\n" + "="*60)
    print("PRÓXIMO PASSO:")
    print("="*60)
    print("1. Use o arquivo gerado no lugar de 'input.txt'")
    print("2. Rode o script de treinamento melhorado")
    print("3. Aguarde o treinamento (pode levar horas)")
    print("="*60)