train_naacl_to_txt:
	python naacl_to_text.py -i train.naacl -o train.txt -t txt

validation_naacl_to_txt:
	python naacl_to_text.py -i validation.naacl -o validation.txt -t txt

test_naacl_to_txt:
	python naacl_to_text.py -i test.naacl -o test.txt -t txt

train_shuffle:
	python naacl_to_text.py -i train.txt -o train_src.txt -t shuffle

test_shuffle:
	python naacl_to_text.py -i test.txt -o test_src.txt -t shuffle

validation_shuffle:
	python naacl_to_text.py -i validation.txt -o validation_src.txt -t shuffle
