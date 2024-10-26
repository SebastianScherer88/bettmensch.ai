## VARIABLES
EXTRAS?=pipelines# see sdk's setup.py for all options
SUITE?=unit# unit,integration,k8s
PYTEST_FLAGS?= # e.g. "-m standard", "-m ddp" for k8s suite appended to the `addopts` setting of the pytest.ini
sdk.install:
	@echo "::group::Installing sdk with $(EXTRAS) extras"
	pip install ./sdk[$(EXTRAS)]
	@echo "::endgroup::"

sdk.test:
	@echo "::group::Running sdk $(SUITE) test suite"
	@echo "::group:: Commit: $(COMMIT)"
	pytest ./sdk/test/$(SUITE)/ $(PYTEST_FLAGS)
	@echo "::endgroup::"