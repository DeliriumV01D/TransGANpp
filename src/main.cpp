#if defined(_MSC_VER)
#define DISABLE_WARNING_PUSH           __pragma(warning( push ))
#define DISABLE_WARNING_POP            __pragma(warning( pop )) 
#define DISABLE_WARNING(warningNumber) __pragma(warning( disable : warningNumber ))
#elif defined(__GNUC__) || defined(__clang__)
#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING_PUSH           DO_PRAGMA(GCC diagnostic push)
#define DISABLE_WARNING_POP            DO_PRAGMA(GCC diagnostic pop) 
#define DISABLE_WARNING(warningName)   DO_PRAGMA(GCC diagnostic ignored #warningName)
#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
#endif

DISABLE_WARNING_PUSH

#if defined(_MSC_VER)
DISABLE_WARNING(4624)
#endif

#include <torch/torch.h>

DISABLE_WARNING_POP

#include <cmath>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

// The size of the noise vector fed to the generator.
const int64_t NOISE_SIZE = 100;

// The batch size for training.
const int64_t BATCH_SIZE = 64;

// The number of epochs to train.
const int64_t NUMBER_OF_EPOCHS = 120;

// Where to find the MNIST dataset.
const std::string DATA_FOLDER = "../mnist";

// After how many batches to create a new checkpoint periodically.
const int64_t CHECKPOINT_EVERY = 10000;

// How many images to sample at every checkpoint.
const int64_t NUMBER_OF_SAMPLES_PER_CHECKPOINT = 10;

// Set to `true` to restore models and optimizers from previously saved checkpoints.
const bool RESTORE_FROM_CHECKPOINT = false;

// After how many batches to log a new update with the loss value.
const int64_t LOG_INTERVAL = 10;


struct PositionalEncodingImpl : torch::nn::Module
{
	//torch::nn::Dropout dropout;
	torch::Tensor pe;

	PositionalEncodingImpl(const std::string &module_name = "positional_encoding", int d_model = 0, float dropout_value = 0.1, int max_len = 5000)
		: torch::nn::Module(module_name)
	{
		pe = torch::zeros({ 1, max_len, d_model });
		for (int i = 0; i < max_len; i++)
			for (int j = 0; j < d_model; j++)
			{
				if (j % 2 == 0)
					pe[0][i][j] = sin(i * exp(int(j / 2) * 2 * (-log(10000.0) / d_model)));
				else
					pe[0][i][j] = cos(i * exp(int(j / 2) * 2 * (-log(10000.0) / d_model)));
			}

		register_buffer(module_name + "_pe", pe);
	}
	
	//у меня [batch_size, seq_len, emb_dim]
	torch::Tensor forward(torch::Tensor x)
	{
		//[1, 5000, 100]
		auto peb = pe.index({ torch::indexing::Slice(0, torch::indexing::None), torch::indexing::Slice(0, x.size(1)), torch::indexing::Slice(0, torch::indexing::None) });
		//[1, 4, 100]
		peb = peb.repeat({ x.sizes()[0], 1, 1 });
		//[64, 1, 100]
		x = x + peb;
		return x;
	}
};

TORCH_MODULE(PositionalEncoding);

struct TransGANppGeneratorImpl : torch::nn::Module
{
	torch::nn::Linear input;
	PositionalEncoding pos_encoder;
	torch::nn::TransformerEncoderLayer transformer_encoder_layers;
	torch::nn::TransformerEncoder transformer_encoder;
	torch::nn::ConvTranspose2d patch_decoder;

	TransGANppGeneratorImpl(
		const std::string &module_name = "generator", 
		int kNoiseSize = 100,		//Для упрощения размер шумового вектора беру совпадающим со входом траннсформера
		int patch_size = 16,
		float pos_drop_rate = 0.1,
		int n_head = 4,
		float enc_drop_rate = 0.1,
		int n_layers = 3
	)	: torch::nn::Module(module_name),
			input(torch::nn::LinearOptions(kNoiseSize, kNoiseSize * 4/*длина последовательности для patch_size = 16 и картинок 28 на 28*/).bias(false)),
			pos_encoder(module_name + "_positional_encoding", kNoiseSize, pos_drop_rate),
			transformer_encoder_layers(torch::nn::TransformerEncoderLayerOptions(kNoiseSize, n_head).dim_feedforward(256/*default 2048*/).dropout(enc_drop_rate)),
			transformer_encoder(transformer_encoder_layers, n_layers),
			patch_decoder(torch::nn::ConvTranspose2dOptions( kNoiseSize, 1, patch_size).stride(patch_size).padding(2).bias(false))		//100, 4, 4 -> 1*28*28
	{
		register_module(module_name + "_input", input);
		register_module(module_name + "_pos_encoder", pos_encoder);
		register_module(module_name + "_transformer_encoder_layers", transformer_encoder_layers);
		register_module(module_name + "_transformer_encoder", transformer_encoder);
		register_module(module_name + "_patch_decoder", patch_decoder);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		//[64, 100, 1, 1]
		x = torch::relu(input(x.flatten(1)));
		x = x.reshape({x.sizes()[0], 4, x.sizes()[1] / 4});
		//[64, 400] -> [64, 4, 100]
		x = pos_encoder(x);
		//[64, 4, 100]
		x = transformer_encoder(x);
		//[64, 4, 100]
		x = x.permute({ 0, 2, 1 });
		//[64, 100, 4]
		x = x.reshape({ x.sizes()[0], x.sizes()[1], x.sizes()[2] / 2, x.sizes()[2] / 2 });
		//[64, 100, 2, 2]
		x = torch::tanh(patch_decoder(x));
		//[64, 1, 28, 28]
		return x;
	}
};

TORCH_MODULE(TransGANppGenerator);

struct TransGANppDiscriminatorImpl : torch::nn::Module
{
	torch::nn::Conv2d patch_embed;
	PositionalEncoding pos_encoder;
	torch::nn::TransformerEncoderLayer transformer_encoder_layers;
	torch::nn::TransformerEncoder transformer_encoder;
	torch::nn::Linear head;
	//self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

	TransGANppDiscriminatorImpl(
		const std::string &module_name = "discriminator",
		int embed_dim = 100,
		int patch_size = 16,
		float pos_drop_rate = 0.1,
		int n_head = 4,
		float enc_drop_rate = 0.1,
		int n_layers = 3
	)	: torch::nn::Module(module_name),
			patch_embed(torch::nn::Conv2dOptions(1, embed_dim, patch_size).stride(patch_size).padding(2/*6*/).bias(false)),		//паддинг чтоб с одной картинки собрать последовательность токенов длины 4
			pos_encoder(module_name + "_positional_encoding", embed_dim, pos_drop_rate),
			transformer_encoder_layers(torch::nn::TransformerEncoderLayerOptions(embed_dim, n_head).dim_feedforward(256/*default 2048*/).dropout(enc_drop_rate)),
			//transformer_encoder(torch::nn::TransformerEncoderOptions(transformer_encoder_layers, n_layers).norm(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ 2 })))),		
			transformer_encoder(transformer_encoder_layers, n_layers),
			head(torch::nn::LinearOptions(embed_dim * 4/*длина последовательности для patch_size = 16 и картинок 28 на 28*/, 1).bias(false))
	{
		register_module(module_name + "_patch_embed", patch_embed);
		register_module(module_name + "_pos_encoder", pos_encoder);
		register_module(module_name + "_transformer_encoder_layers", transformer_encoder_layers);
		register_module(module_name + "_transformer_encoder", transformer_encoder);
		register_module(module_name + "_head", head);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		//[64, 1, 28, 28]
		x = patch_embed(x).flatten(2).permute({ 0, 2, 1 });   //[64, 100, 2, 2] -> [64, 100, 4] -> [64, 4, 100]
		//[64, 4, 100]
		//x = x * math.sqrt(self.d_model);
		x = pos_encoder(x);
		//[64, 4, 100]
		x = transformer_encoder(x);
		//Какой-то выход есть теперь привести к нужному виду головой - линейным слоем
		x = torch::sigmoid(head(x.flatten(1)));  //[64, 4, 100] -> [64, 400] -> [64, 1]
		return x;
		//[64, 1, 1, 1]
	}
};

TORCH_MODULE(TransGANppDiscriminator);

void Visualize(const torch::Tensor &samples)
{
	int n = 10;
	cv::Mat scene(cv::Size(samples.sizes()[2] * n, samples.sizes()[3]), CV_32F);
	
	for (int i = 0; i < n; i++)
	{
		//64 1 28 28 -> 1 28 28
		auto image_tensor = samples[i].detach().cpu();
		cv::Mat image_mat(image_tensor.size(1), image_tensor.size(2), CV_32F, image_tensor.data_ptr());
		image_mat.copyTo(scene(cv::Rect(image_mat.cols * i, 0, image_mat.cols, image_mat.rows)));
	}
	cv::imshow("visualize", scene);
	cv::waitKey(1);
}

int main(int argc, const char* argv[])
{
	torch::manual_seed(42);

	// Create the device we pass around based on whether CUDA is available.
	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available()) 
	{
		std::cout << "CUDA is available! Training on GPU." << std::endl;
		device = torch::Device(torch::kCUDA);
	} else 
	{
		std::cout << "CUDA is not available! Training on CPU." << std::endl;
	}

	TransGANppGenerator generator("generator", NOISE_SIZE);
	generator->to(device);

	std::cout << "generator:" << std::endl;
	for (auto k : generator->named_parameters())
		std::cout << k.key() << std::endl;

	TransGANppDiscriminator discriminator("discriminator");
	discriminator->to(device);

	std::cout << "discriminator:" << std::endl;
	for (auto k : discriminator->named_parameters())
		std::cout << k.key() << std::endl;

	// Assume the MNIST dataset is available under `kDataFolder`;
	auto dataset = torch::data::datasets::MNIST(DATA_FOLDER)
		.map(torch::data::transforms::Normalize<>(0.5, 0.5))
		.map(torch::data::transforms::Stack<>());
	const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(BATCH_SIZE));

	auto data_loader = torch::data::make_data_loader(
		std::move(dataset),
		torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(2)
	);

	torch::optim::Adam generator_optimizer(generator->parameters(), torch::optim::AdamOptions(1e-4).weight_decay(0.001).betas(std::make_tuple (0.5, 0.99)));
	torch::optim::Adam discriminator_optimizer(discriminator->parameters(), torch::optim::AdamOptions(2e-4).weight_decay(0.001).betas(std::make_tuple(0.5, 0.99)));

	torch::optim::StepLR generator_sheduler(generator_optimizer, 30, 0.1);
	torch::optim::StepLR discriminator_sheduler(discriminator_optimizer, 30, 0.1);

	auto params_count = [](auto module_) 
	{
		int result = 0; 
		for (auto p : module_->parameters())
		{
			int ss = 1;
			for (auto s : p.sizes())
				ss *= s;
			result += ss;
		}
		return result;
	};
	std::cout << "generator parameters count: " << params_count(generator) << std::endl;
	std::cout << "discriminator parameters count: " << params_count(discriminator) << std::endl;

	if (RESTORE_FROM_CHECKPOINT) 
	{
		std::cout << "restoring parameters from checkpoint..." << std::endl;
		torch::load(generator, "generator-checkpoint.pt");
		torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
		torch::load(discriminator, "discriminator-checkpoint.pt");
		torch::load(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
	}	else {
		std::cout << "initializing parameters with xavier..." << std::endl;
		for (auto &p : generator->named_parameters())
		{
			if (p.key().find("norm") != p.key().npos && p.key().find(".weight") != p.key().npos)
			{
				std::cout << p.key() << std::endl;
				generator->named_parameters()[p.key()] = torch::nn::init::constant_(p.value(), 1.);
			} else 	if (p.key().find(".weight") != p.key().npos) 
			{
				std::cout << p.key() << std::endl;
				generator->named_parameters()[p.key()] = torch::nn::init::xavier_normal_(p.value(), 0.1);
			}

			if (p.key().find(".bias") != p.key().npos)
			{
				std::cout << p.key() << std::endl;
				generator->named_parameters()[p.key()] = torch::nn::init::constant_(p.value(), 0.);
			}
		}

		for (auto &p : discriminator->named_parameters())
		{
			if (p.key().find("norm") != p.key().npos && p.key().find(".weight") != p.key().npos)
			{
				std::cout << p.key() << std::endl;
				discriminator->named_parameters()[p.key()] = torch::nn::init::constant_(p.value(), 1.);
			} else if (p.key().find(".weight") != p.key().npos)
			{
				std::cout << p.key() << std::endl;
				discriminator->named_parameters()[p.key()] = torch::nn::init::xavier_normal_(p.value(), 0.1);
			}

			if (p.key().find(".bias") != p.key().npos)
			{
				std::cout << p.key() << std::endl;
				discriminator->named_parameters()[p.key()] = torch::nn::init::constant_(p.value(), 0);
			}
		}
	}

	generator->train();
	discriminator->train();

	int64_t checkpoint_counter = 1;
	//train loop из семи залуп
	for (int64_t epoch = 1; epoch <= NUMBER_OF_EPOCHS; epoch++) 
	{
		for (auto &p : discriminator_optimizer.param_groups())
			std::cout << "discriminator optimizer lr = "<< p.options().get_lr() << std::endl;
		for (auto &p : generator_optimizer.param_groups())
			std::cout << "generator optimizer lr = " << p.options().get_lr() << std::endl;
		int64_t batch_index = 0;
		for (torch::data::Example<> &batch : *data_loader) 
		{
			// Train discriminator with real images.
			discriminator->zero_grad();
			torch::Tensor real_images = batch.data.to(device);
			torch::Tensor real_output = discriminator->forward(real_images);
			torch::Tensor real_labels = torch::empty(real_output.sizes(), device).uniform_(0.8, 1.0);//!!!(0.8, 1.0); label smoothing
			torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
			//d_loss_real.backward();

			// Train discriminator with fake images.
			torch::Tensor noise = torch::randn({batch.data.size(0), NOISE_SIZE, 1, 1}, device);
			torch::Tensor fake_images = generator->forward(noise);
			torch::Tensor fake_output = discriminator->forward(fake_images.detach());
			torch::Tensor fake_labels = torch::zeros(fake_output.sizes(), device);
			torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
			//d_loss_fake.backward();

			//Update discriminator weights
			torch::Tensor d_loss = d_loss_real + d_loss_fake;
			d_loss.backward();
			discriminator_optimizer.step();

			// Train generator.
			generator->zero_grad();

			//Generate new fake images
			noise = torch::randn({ batch.data.size(0), NOISE_SIZE, 1, 1 }, device);
			fake_images = generator->forward(noise);
			Visualize(fake_images);
			
			//Try to fool the discriminator
			fake_labels.fill_(1);
			fake_output = discriminator->forward(fake_images);
			torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);

			//попускаем улучшение генератора, пока дискриминатор не вернет лидерство
			if (d_loss.item<float>() <= 0.5)
			{
				g_loss.backward();
				generator_optimizer.step();
			}
			
			batch_index++;

			if (batch_index % LOG_INTERVAL == 0) 
			{
				std::cout << "[" << epoch << "|" << NUMBER_OF_EPOCHS << "][" << batch_index << "|" << batches_per_epoch << "] d_loss: " << d_loss.item<float>() << "; g_loss: " << g_loss.item<float>() <<";" << std::endl;
			}

			if (batch_index % CHECKPOINT_EVERY == 0) 
			{
				// Checkpoint the model and optimizer state.
				torch::save(generator, "generator-checkpoint.pt");
				torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
				torch::save(discriminator, "discriminator-checkpoint.pt");
				torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
				std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
			}
		}
		generator_sheduler.step();
		discriminator_sheduler.step();
	}

	std::cout << "Training complete!" << std::endl;
	cv::waitKey(0);
}