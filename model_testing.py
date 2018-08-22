import sentence_classification as sc
#testing on unlabeled dataset
jdm_path = "/home/ubuntu/SentenceCorpus/unlabeled_articles/jdm_unlabeled"
jdm_text_files = sc.txt_file(jdm_path)
jdm_s = []
for text_file in jdm_text_files:
    f = open(text_file, "r")
    lines = f.readlines()
    for sent in lines:
        if sent.startswith("#"):
            continue
        elif sent.endswith("\n"):
            s = sent[:-1]
        else:
            s = sent
        jdm_s.append(s)

jdm_s = jdm_s[0:5000]  #choosing random 5000 unlabeled sentence for test

jdm_data = []
for i in range(len(jdm_s)):
    predicted_data = sc.nb.classify(jdm_s[i]).upper()
    jdm_l = predicted_data
    d = {'label': jdm_l, "text": jdm_s[i] }
    jdm_data.append(d)

jdm_txt = "/home/ubuntu/SentenceCorpus" + "/" + "jdm_labeled" + ".txt"
with open(jdm_txt, 'w') as f:
    f.write(str(jdm_data))
