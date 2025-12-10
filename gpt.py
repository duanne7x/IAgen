import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import gc

# ========== CONFIGURAÇÃO PARA CELULAR COM 12GB RAM ==========
# Modelo MÉDIO que usa ~2-3GB RAM - MUITO MAIS PODEROSO!
batch_size = 48         # AUMENTADO (processa mais dados por vez)
block_size = 256        # AUMENTADO (entende textos maiores)
max_iters = 10000       # AUMENTADO (treina por mais tempo)
eval_interval = 500
learning_rate = 3e-4
device = 'cpu'
eval_iters = 100        # AUMENTADO (avalia melhor)

# Arquitetura MÉDIA (~15M parâmetros)
n_embd = 256           # AUMENTADO (128 -> 256)
n_head = 8             # AUMENTADO (4 -> 8)
n_layer = 6            # AUMENTADO (4 -> 6)
dropout = 0.2

# Gradient accumulation (simula batch maior sem usar mais RAM de uma vez)
gradient_accumulation_steps = 4

print("="*60)
print("🚀 GPT MÉDIO - OTIMIZADO PARA 12GB RAM")
print("="*60)
print(f"📱 Dispositivo: {device.upper()}")
print(f"💾 RAM que será usada: ~2-3GB")
print(f"🧠 Modelo: {n_layer} camadas, {n_embd} dimensões, {n_head} heads")
print(f"⏱️  Tempo estimado: 45min - 2h")
print(f"📊 Batch size: {batch_size} x {gradient_accumulation_steps} acumulação")
print("="*60)
# ------------

torch.manual_seed(1337)

# Carregar dados
print("\n📂 Carregando dataset...")
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"✓ Dataset carregado: {len(text):,} caracteres")
    print(f"  ({len(text)/1024:.1f}KB)")
except FileNotFoundError:
    print("❌ ERRO: Arquivo 'input.txt' não encontrado!")
    print("   Coloque seu dataset na mesma pasta do script.")
    exit(1)

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"✓ Vocabulário: {vocab_size} caracteres únicos")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """Uma cabeça de self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scaled attention (IMPORTANTE para estabilidade)
        wei = q @ k.transpose(-2,-1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Múltiplas cabeças de atenção"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Rede feed-forward com ReLU"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Bloco Transformer com Pre-LayerNorm"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LayerNorm (mais estável)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Weight tying (compartilhar pesos economiza memória)
        self.token_embedding_table.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Gera texto com temperatura e top-k sampling"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling (melhor qualidade)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Criar modelo
print("\n🔨 Criando modelo...")
model = GPTLanguageModel()
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Modelo criado: {total_params/1e6:.2f}M parâmetros")
print(f"  Tamanho aproximado: {total_params * 4 / 1024 / 1024:.1f}MB")

# Optimizer com weight decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate,
    weight_decay=0.01
)

# Learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=learning_rate/10)

# Training
print("\n🚀 Iniciando treinamento...")
print("   (Usando gradient accumulation para simular batch maior)")
print("="*60)

start_time = time.time()
best_val_loss = float('inf')

for iter in range(max_iters):

    # Avaliação periódica
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        steps_per_sec = (iter + 1) / elapsed
        eta = (max_iters - iter - 1) / steps_per_sec if steps_per_sec > 0 else 0
        
        print(f"step {iter:5d}/{max_iters} | "
              f"train {losses['train']:.4f} | "
              f"val {losses['val']:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed/60:.1f}min | "
              f"ETA ~{eta/60:.1f}min")
        
        # Salvar melhor modelo
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"   💾 Novo melhor modelo salvo! (val loss: {best_val_loss:.4f})")
        
        # Checkpoint periódico
        if iter > 0 and iter % 2000 == 0:
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': losses['train'],
                'val_loss': losses['val'],
            }, f'checkpoint_{iter}.pth')
            print(f"   💾 Checkpoint salvo: checkpoint_{iter}.pth")

    # Training step com gradient accumulation
    model.train()
    optimizer.zero_grad()
    
    # Acumular gradientes
    total_loss = 0
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        loss = loss / gradient_accumulation_steps  # Normalizar
        loss.backward()
        total_loss += loss.item()
    
    # Atualizar pesos
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
    optimizer.step()
    scheduler.step()
    
    # Limpar memória periodicamente
    if iter % 100 == 0:
        gc.collect()

# Salvar modelo final
print("\n" + "="*60)
print("💾 Salvando modelo treinado...")
torch.save(model.state_dict(), 'modelo_final.pth')
print("✓ Modelo final salvo em 'modelo_final.pth'")
print("✓ Melhor modelo salvo em 'best_model.pth'")

# Carregar melhor modelo para geração
print("\n📥 Carregando melhor modelo para gerar texto...")
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Gerar texto com diferentes temperaturas
print("\n" + "="*60)
print("📝 EXEMPLOS DE TEXTO GERADO:")
print("="*60)

temperatures = [0.8, 1.0, 1.2]
for temp in temperatures:
    print(f"\n--- Temperatura {temp} (menor = conservador, maior = criativo) ---")
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=200, temperature=temp, top_k=40)
    print(decode(generated[0].tolist()))
    print()

print("="*60)
print("✅ TREINAMENTO CONCLUÍDO!")
print(f"⏱️  Tempo total: {(time.time() - start_time)/60:.1f} minutos")
print(f"📊 Melhor val loss: {best_val_loss:.4f}")
print("="*60)
print("\n💡 PRÓXIMOS PASSOS:")
print("   1. Se texto ficou bom: ótimo! Modelo treinado")
print("   2. Se texto ainda ruim: treine por mais tempo (aumente max_iters)")
print("   3. Para usar português: adicione textos em PT ao input.txt")
print("   4. Modelo salvo em 'best_model.pth' (melhor) e 'modelo_final.pth'")
print("="*60)
