type DependencyType = 
  | "method_call"      // @Autowired, direct invocation
  | "http"            // RestTemplate/WebClient
  | "feign_client"    // @FeignClient
  | "database"        // JPA/JDBC
  | "message_queue"   // Kafka/RabbitMQ
  | "inheritance"     // extends/implements