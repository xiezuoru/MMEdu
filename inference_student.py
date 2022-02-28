from utils.my_utils_new import MMClassification


def test1():
	img = 'utils/demo/bird.JPEG'
	model = MMClassification()
	result = model.inference(image=img)
	print(result)


def test2():
	img = 'utils/demo/bird.JPEG'
	model = MMClassification(pretrain='MobileNet', checkpoints='')
	result = model.inference(image=img)
	print(result)


def test3():
	model = MMClassification()
	model.load_dataset(path='cats_dogs_dataset', dataset_type='ImageNet')
	model.train(epochs=20, device='cuda:0', validate=False, backbone="MobileNet")
	model.inference(is_trained=True, image='./cats_dogs_dataset/test_set/test_set/cats/cat.4003.jpg')


def test4():
	model = MMClassification()
	model.load_dataset(path='data/mnist', dataset_type='ImageNet')
	model.train(epochs=1, device='cpu', validate=False, backbone="LeNet")
	# model.inference(is_trained=True, image='./cats_dogs_dataset/test_set/test_set/cats/cat.4003.jpg')


if __name__ == "__main__":
	# test1()
	# test2()
	# test3()
	test4()
