from embeddings import Embeddings, PatchEmbeddings

class ViT():
  def _init_(self):
    self.img_size = (224,224)
    self.patch_size = (16,16)

    self.embeddings = PatchEmbeddings()

  def forward(self,x):
    x = self.embeddings(x)
    return x