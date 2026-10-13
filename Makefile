BUILD_DIR ?= build

.PHONY: clangformat

clangformat:
	@if [ ! -f "$(BUILD_DIR)/CMakeCache.txt" ]; then \
		echo "Build directory '$(BUILD_DIR)' is not configured."; \
		echo "Run cmake -S . -B $(BUILD_DIR) -DAUTOPAS_SOURCE_DIR=../AutoPas first."; \
		exit 1; \
	fi
	@cmake --build "$(BUILD_DIR)" --target autopasdm_clangformat
