from luna16 import services


def test_service_container() -> None:
    registry = services.ServiceContainer()
    registry.register_service(str, value="Test")

    assert len(registry.services) == 1
    assert registry.get_service(str) == "Test"
