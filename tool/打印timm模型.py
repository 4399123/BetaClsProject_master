import timm

models=timm.list_models(pretrained=True)
# models=timm.list_models()
for model in models:
    print(model)