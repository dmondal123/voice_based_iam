package com.griddynamics.capstone_project.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

@Service
public class DetectionService {

    @Value("${fastapi.url}")
    private String fastApiUrl;

    private final WebClient webClient;

    public DetectionService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.baseUrl(fastApiUrl).build();
    }

    public String processLogFile(MultipartFile file) throws IOException {
        Path tempFile = Files.createTempFile("application", ".log");
        Files.copy(file.getInputStream(), tempFile, StandardCopyOption.REPLACE_EXISTING);

        try {
            return webClient.post()
                    .uri("/process-log") // detection python fast api ...
                    .body(BodyInserters.fromResource(file.getResource()))
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();
        } catch (WebClientResponseException e) {
            throw new IOException("Error communicating with py DS detection service: " + e.getMessage(), e);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }
}
