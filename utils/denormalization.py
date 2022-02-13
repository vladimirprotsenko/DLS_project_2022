def denorm(img_to_denorm):
		stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
		return img_to_denorm * stats[1][0] + stats[0][0]