package com.griddynamics.capstone_project.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.PropertySource;
import org.springframework.lang.NonNull;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;

@Service
@PropertySource("classpath:auth.properties")
public class AudioAuthService {
    private final WebClient webClient;

//    @Value("${base}")
//    private String fastAPIBaseUrl ;

    public AudioAuthService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.baseUrl("http://localhost:8000").build(); //fastAPI Bbase url
    }

    public String predictUser(@NonNull MultipartFile file) throws Exception { //enable better exception handling
        return webClient.post()
                .uri("/predict/")
                .body(BodyInserters.fromMultipartData("file", file.getResource()))
                .retrieve()
                .bodyToMono(String.class)
                .block();
    }
}

