from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(filename="julius-caesar_PDF_FolgerShakespeare.pdf")
for el in elements:
    print(type(el), getattr(el, "text", None))