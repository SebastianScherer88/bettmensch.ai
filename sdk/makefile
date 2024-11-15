## VARIABLES
UNINSTALL?=true
EXTRAS?=pipelines# see sdk's setup.py for all options
SUITE?=unit# unit,integration,k8s
PYTEST_FLAGS?= # e.g. "-m standard", "-m ddp" for k8s suite appended to the `addopts` setting of the pytest.ini
sdk.install:
	@echo "::group::Installing sdk with $(EXTRAS) extras"
	if [ $(UNINSTALL)=true ]; then \
		make sdk.uninstall; \
	fi;\
	pip install ./sdk[$(EXTRAS)] --no-cache-dir
	@echo "::endgroup::"

sdk.test:
	@echo "::group::Running sdk $(SUITE) test suite"
	@echo "::group:: Commit: $(COMMIT)"
	pytest ./sdk/test/$(SUITE)/ $(PYTEST_FLAGS) > ./sdk/test/$(SUITE)/outputs/pytest.out
	@echo "::endgroup::"

sdk.uninstall:
	@echo "::group::Uninstalling sdk with $(EXTRAS) extras"
	pip uninstall bettmensch-ai -y
	@echo "::endgroup::"