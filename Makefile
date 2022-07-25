
threads?=2

psdd_sample:
	python main.py --tmode full --net mnist --threads $(threads) --cir_type psdd

sptrsv_sample:
	python main.py --tmode full --net HB/bcspwr01 --threads $(threads) --cir_type sptrsv

all:
	python main.py --tmode full --net HB/bcspwr01 --threads $(threads) --cir_type sptrsv
	python main.py --tmode full --net HB/bcsstm02 --threads $(threads) --cir_type sptrsv
	python main.py --tmode full --net HB/bcsstm05 --threads $(threads) --cir_type sptrsv
	python main.py --tmode full --net HB/bcsstm22 --threads $(threads) --cir_type sptrsv
	python main.py --tmode full --net HB/can_24  --threads $(threads) --cir_type sptrsv
	python main.py --tmode full --net HB/can_62  --threads $(threads) --cir_type sptrsv
	python main.py --tmode full --net HB/ibm32   --threads $(threads) --cir_type sptrsv
	python main.py --tmode full --net tretail    --threads $(threads) --cir_type psdd
	python main.py --tmode full --net mnist      --threads $(threads) --cir_type psdd
	python main.py --tmode full --net nltcs      --threads $(threads) --cir_type psdd
	python main.py --tmode full --net kdd        --threads $(threads) --cir_type psdd
	python main.py --tmode full --net msnbc      --threads $(threads) --cir_type psdd
	python main.py --tmode full --net msweb      --threads $(threads) --cir_type psdd
	python main.py --tmode full --net ad         --threads $(threads) --cir_type psdd
	python main.py --tmode full --net baudio     --threads $(threads) --cir_type psdd
	python main.py --tmode full --net bbc        --threads $(threads) --cir_type psdd
	python main.py --tmode full --net bnetflix   --threads $(threads) --cir_type psdd
	python main.py --tmode full --net book       --threads $(threads) --cir_type psdd
	python main.py --tmode full --net c20ng      --threads $(threads) --cir_type psdd
	python main.py --tmode full --net cr52       --threads $(threads) --cir_type psdd
	python main.py --tmode full --net cwebkb     --threads $(threads) --cir_type psdd
	python main.py --tmode full --net jester     --threads $(threads) --cir_type psdd

